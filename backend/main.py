"""
Court File Indexer — RAG Backend
FastAPI server handling:
  - PDF ingestion (OCR + vectorization)
  - Semantic search / chatbot queries
  - Automatic index generation
  - Local LLM reasoning and vision assistance
"""

import os
import asyncio
import re
import json
import math
import base64
import hashlib
import logging
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from io import BytesIO
from threading import Lock, Thread
from typing import Optional

import fitz                        # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import httpx

from workflow_state import (
    STORAGE_BACKEND as WORKFLOW_STORAGE_BACKEND,
    STORAGE_TARGET as WORKFLOW_STORAGE_TARGET,
    build_queue_snapshot,
    delete_pdf_state,
    get_cached_pages,
    get_pdf_record,
    get_saved_index,
    init_db as init_workflow_db,
    list_pdf_records,
    list_pending_pdf_ids,
    replace_extracted_pages,
    save_index,
    update_pdf_record,
    upsert_extracted_pages,
    upsert_pdf_record,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_CACHE_ROOT = Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir()) / "court-rag-hf-cache"
HF_CACHE_PATH = str(HF_CACHE_ROOT)
os.environ.setdefault("HF_HOME", HF_CACHE_PATH)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_PATH)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
Path(HF_CACHE_PATH).mkdir(parents=True, exist_ok=True)
DOCUMENT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "document_catalog.json"

LOCAL_LLM_BASE_URL = (os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
LOCAL_TEXT_MODEL = os.getenv("LOCAL_TEXT_MODEL", "qwen2.5:14b")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "180"))
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "./stored_pdfs")
INDEX_EXPORT_PATH = os.getenv("INDEX_EXPORT_PATH", "./index_exports")
TESSERACT_LANG   = os.getenv("TESSERACT_LANG", "hin+eng")
ENABLE_HANDWRITTEN_HINDI_ASSIST = os.getenv("ENABLE_HANDWRITTEN_HINDI_ASSIST", "true").lower() != "false"
try:
    PARENT_DOCUMENT_CATALOG = json.loads(DOCUMENT_CATALOG_PATH.read_text(encoding="utf-8"))
except Exception:
    PARENT_DOCUMENT_CATALOG = []
PARENT_DOCUMENT_NAMES = list(dict.fromkeys(
    item["name"].strip()
    for item in PARENT_DOCUMENT_CATALOG
    if item.get("name") and str(item["name"]).strip()
))
PARENT_DOCUMENT_EMBEDDINGS = None
GENERIC_PARENT_NAMES = {"other", "others"}
STRUCTURAL_TOC_PATTERNS = [
    r"index",
    r"table of contents",
    r"contents",
    r"list of documents",
    r"chronological",
    r"chronology",
    r"events of the case",
    r"memo of civil revision",
    r"memo",
    r"part\s*[a-z0-9-]+",
]
NEGATIVE_TOC_MAPPINGS = {
    "memo of civil revision": {"Index"},
    "list of documents": {"Others", "Other"},
    "chronological": {"Vakalat Nama", "Reply", "Application"},
    "events of the case": {"Vakalat Nama", "Reply", "Application"},
}
LOW_CONFIDENCE_TOC_SOURCES = {"toc", "toc-image"}
TOC_TABLE_HEADER_PATTERNS = [
    r"\bs\.?\s*no\b",
    r"\bserial\s*no\b",
    r"\bdescription\b",
    r"\bdocument\b",
    r"\bannexure\b",
    r"\bannx\b",
    r"\bpages?\b",
    r"\bpage\s*no\b",
    r"\bsheets?\b",
    r"\bparticulars?\b",
]

# Local LLM configuration

# ── Embedding model (local, offline, Hindi+English) ───────────────────────────
embedder = None
embedder_lock = Lock()

deferred_runner_lock = Lock()
deferred_runner_status = {
    "running": False,
    "processed": 0,
    "total": 0,
    "current_pdf_id": "",
    "current_filename": "",
    "last_error": "",
    "heartbeat_ts": time.time(),
    "pause_requested": False,
    "paused": False,
}

index_runner_lock = Lock()
index_runner_status = {
    "running": False,
    "current_pdf_id": "",
    "current_filename": "",
    "last_error": "",
    "heartbeat_ts": time.time(),
    "started_at": 0.0,
    "finished_pdf_id": "",
    "finished_filename": "",
    "status": "idle",
}

# ── ChromaDB ──────────────────────────────────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(INDEX_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Court File Indexer API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════


TIMING_SUMMARY_ORDER = [
    "file_open",
    "first_10_page_extraction",
    "ocr_time",
    "toc_detection",
    "llm_indexing_time",
    "json_generation_time",
    "full_text_extraction_time",
    "chunking_time",
    "embedding_time",
    "vector_db_insert_time",
    "total_vectorization_time",
]

TIMING_LABELS = {
    "file_open": "file open",
    "first_10_page_extraction": "first 10-page extraction",
    "ocr_time": "OCR time",
    "toc_detection": "TOC detection",
    "llm_indexing_time": "LLM indexing time",
    "json_generation_time": "JSON generation time",
    "full_text_extraction_time": "full text extraction time",
    "chunking_time": "chunking time",
    "embedding_time": "embedding time",
    "vector_db_insert_time": "vector DB insert time",
    "total_vectorization_time": "total vectorization time",
}


class StageTimer:
    def __init__(self, name: str, collector: Optional["PdfTimingCollector"] = None, summary_key: str = ""):
        self.name = name
        self.collector = collector
        self.summary_key = summary_key
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        log.info("[START] %s", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        if self.collector and self.summary_key:
            self.collector.add_duration(self.summary_key, elapsed)
        log.info("[END] %s took %.3fs", self.name, elapsed)


class PdfTimingCollector:
    def __init__(self, pdf_id: str, filename: str = ""):
        self.pdf_id = pdf_id
        self.filename = filename or ""
        self.timings: dict[str, float] = {}

    def stage(self, name: str, summary_key: str) -> StageTimer:
        return StageTimer(name, collector=self, summary_key=summary_key)

    def add_duration(self, key: str, elapsed: float):
        self.timings[key] = self.timings.get(key, 0.0) + elapsed

    def log_summary(self, run_label: str):
        if not self.timings:
            return
        log.info(
            "[PDF TIMING] Summary for pdf=%s file=%s run=%s",
            self.pdf_id,
            self.filename or "-",
            run_label,
        )
        for key in TIMING_SUMMARY_ORDER:
            if key in self.timings:
                log.info(
                    "[PDF TIMING] pdf=%s %s = %.3fs",
                    self.pdf_id,
                    TIMING_LABELS.get(key, key),
                    self.timings[key],
                )

def pdf_id_from_bytes(data: bytes) -> str:
    """Stable ID for a PDF based on its content hash."""
    return hashlib.md5(data).hexdigest()[:16]


def stored_pdf_path(pdf_id: str) -> Path:
    """Local on-disk copy of the uploaded PDF for later page-image reprocessing."""
    return Path(PDF_STORAGE_PATH) / f"{pdf_id}.pdf"


def sanitize_export_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\|?*\x00-\x1f]+', "_", (value or "").strip())
    return cleaned.strip(" .") or "index"


def export_index_json(
    pdf_id: str,
    record: Optional[dict],
    index_items: list[dict],
    indexed_start: int,
    indexed_end: int,
    total_pages: int,
    index_source: str,
) -> Path:
    cnr_number = (
        (record or {}).get("cnr_number")
        or cnr_number_from_filename((record or {}).get("filename", ""))
        or pdf_id
    )
    payload = {
        "pdf_id": pdf_id,
        "cnr_number": cnr_number,
        "file_number": cnr_number,
        "filename": (record or {}).get("filename", ""),
        "total_pages": total_pages,
        "indexed_page_start": indexed_start,
        "indexed_page_end": indexed_end,
        "indexed_pages": max(indexed_end - indexed_start + 1, 0),
        "index_source": index_source,
        "status": (record or {}).get("status", "index_ready"),
        "index_entries": len(index_items or []),
        "index": index_items or [],
    }
    export_path = Path(INDEX_EXPORT_PATH) / f"{sanitize_export_stem(cnr_number)}.json"
    export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return export_path


def get_or_create_collection(pdf_id: str):
    """Get or create a ChromaDB collection for this PDF."""
    name = f"pdf_{pdf_id}"
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def get_embedder():
    """Load the embedding model only when vector work is actually needed."""
    global embedder
    if embedder is None:
        with embedder_lock:
            if embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    log.info("Loading embedding model...")
                    embedder = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        cache_folder=HF_CACHE_PATH,
                    )
                    log.info("Embedding model ready")
                except Exception as e:
                    embedder = False
                    log.exception("Falling back to lightweight local embeddings: %s", e)
    return embedder


def fallback_embed_texts(texts: list[str], dims: int = 384) -> list[list[float]]:
    """Simple local embedding fallback when transformer weights cannot load."""
    vectors = []
    for text in texts:
        vec = [0.0] * dims
        tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using the local multilingual model."""
    model = get_embedder()
    if model is False:
        return fallback_embed_texts(texts)
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


def ocr_page_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Returns extracted text."""
    try:
        text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config="--psm 6")
        return text.strip()
    except Exception as e:
        log.warning(f"Tesseract OCR failed: {e}")
        return ""


def render_page_image(page: fitz.Page, dpi: int = 250) -> Image.Image:
    """Render a PDF page to a PIL image for OCR and vision extraction."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_jpeg_base64(image: Image.Image, max_side: int = 1800, quality: int = 80) -> str:
    """Convert a PIL image to base64 JPEG, shrinking large pages for vision OCR."""
    img = image.copy()
    resampling = getattr(Image, "Resampling", Image)
    img.thumbnail((max_side, max_side), resampling.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def analyze_extracted_text(text: str) -> dict:
    content = (text or "").strip()
    if not content:
        return {
            "chars": 0,
            "words": 0,
            "line_count": 0,
            "ascii_ratio": 0.0,
            "devanagari_ratio": 0.0,
            "digit_ratio": 0.0,
        }

    chars = len(content)
    words = len(re.findall(r"\w+", content, flags=re.UNICODE))
    line_count = len([line for line in content.splitlines() if line.strip()])
    devanagari = len(re.findall(r"[\u0900-\u097F]", content))
    ascii_letters = len(re.findall(r"[A-Za-z]", content))
    digits = len(re.findall(r"\d", content))
    return {
        "chars": chars,
        "words": words,
        "line_count": line_count,
        "ascii_ratio": ascii_letters / max(chars, 1),
        "devanagari_ratio": devanagari / max(chars, 1),
        "digit_ratio": digits / max(chars, 1),
    }


def should_try_handwritten_assist(direct_text: str, ocr_text: str) -> bool:
    """Heuristic gate for hard scanned pages where Tesseract is likely insufficient."""
    ocr_stats = analyze_extracted_text(ocr_text)
    direct_stats = analyze_extracted_text(direct_text)
    likely_scan = direct_stats["chars"] < 30
    weak_ocr = ocr_stats["chars"] < 140 or ocr_stats["words"] < 24 or ocr_stats["line_count"] < 4
    mixed_noise = ocr_stats["digit_ratio"] > 0.22 and ocr_stats["words"] < 40
    low_script_signal = ocr_stats["devanagari_ratio"] < 0.02 and ocr_stats["ascii_ratio"] < 0.12
    return likely_scan and (weak_ocr or mixed_noise or low_script_signal)


def extract_handwritten_page_text(image: Image.Image, page_num: int) -> Optional[str]:
    """Use the vision model as a higher-accuracy fallback for handwritten Hindi/English pages."""
    if not ENABLE_HANDWRITTEN_HINDI_ASSIST or not LOCAL_VISION_MODEL:
        return None

    prompt = f"""You are transcribing a scanned Indian court-file page.
This page may contain handwritten Hindi, handwritten English, printed Hindi, printed English, or a mixture.

Read the page as accurately as possible and return only the page text.

Rules:
- Preserve Hindi in Devanagari.
- Preserve English exactly.
- Keep line breaks where helpful.
- Do not summarize, translate, classify, or explain.
- If a word is unclear, make the best reading you can instead of dropping it.
- Include headings, labels, serial numbers, page ranges, names, dates, and table row text if visible.

Return only the transcription for page {page_num}."""
    try:
        image_b64 = image_to_jpeg_base64(image)
        text = call_local_vision(image_b64, "image/jpeg", prompt, max_tokens=2200).strip()
        return text or None
    except Exception as exc:
        log.warning("Vision transcription failed for page %s: %s", page_num, exc)
        return None


def extract_page_content(
    page: fitz.Page,
    page_num: int,
    dpi: int = 200,
    timing_collector: Optional[PdfTimingCollector] = None,
) -> dict:
    """
    Extract text from a PDF page.
    First tries direct text extraction (fast, perfect for digital PDFs).
    Falls back to OCR if the page is a scan.
    For hard handwritten/low-quality Hindi pages, optionally uses the vision model
    to improve the stored text used by indexing and chat.
    """
    direct_text = page.get_text("text").strip()
    if len(direct_text) > 30:
        clean = re.sub(r"\n{3,}", "\n\n", direct_text)
        return {
            "text": clean,
            "used_ocr": False,
            "vision_used": False,
            "handwriting_suspected": False,
            "extraction_method": "digital",
        }

    ocr_started = time.perf_counter()
    image = render_page_image(page, dpi=dpi)
    ocr_text = ocr_page_image(image)
    vision_used = False
    handwriting_suspected = should_try_handwritten_assist(direct_text, ocr_text)
    final_text = ocr_text
    extraction_method = "ocr"

    if handwriting_suspected:
        enhanced_text = extract_handwritten_page_text(image, page_num)
        if enhanced_text:
            enhanced_stats = analyze_extracted_text(enhanced_text)
            ocr_stats = analyze_extracted_text(ocr_text)
            if enhanced_stats["chars"] >= max(ocr_stats["chars"], 80):
                final_text = enhanced_text
                vision_used = True
                extraction_method = "vision_ocr"

    if timing_collector:
        timing_collector.add_duration("ocr_time", time.perf_counter() - ocr_started)

    return {
        "text": final_text,
        "used_ocr": True,
        "vision_used": vision_used,
        "handwriting_suspected": handwriting_suspected,
        "extraction_method": extraction_method,
    }

def extract_toc_from_page_images(pdf_path: Path, page_nums: list[int]) -> list[dict]:
    """Use page images for TOC extraction when a likely Hindi/English index page is found."""
    if not page_nums or not pdf_path.exists() or not LOCAL_VISION_MODEL:
        return []

    toc_items = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        log.warning("Could not open stored PDF for TOC image parsing: %s", exc)
        return []

    try:
        for page_num in page_nums:
            if page_num < 1 or page_num > doc.page_count:
                continue
            image = render_page_image(doc[page_num - 1], dpi=240)
            image_b64 = image_to_jpeg_base64(image, max_side=2200, quality=84)
            prompt = f"""You are reading a scanned Indian court-file Table of Contents / Index page.
The page may be in English, Hindi (Devanagari), or mixed text, and may include handwritten entries.

Task:
- Decide whether this page is a real TOC / index / ???? / ???? ???? / ??????????? page.
- If it is, extract every readable table row from the page image.
- Focus on table rows only, not the case title or header text above the table.
- Read table columns carefully: serial number, description, annexure, pages, sheet count, court fee.
- Preserve the row description exactly as written, especially Hindi.
- Convert Hindi digits to Arabic numerals in pageFrom/pageTo if needed.
- If a row spans one page, set pageFrom and pageTo to the same value.
- If a row title continues on the next line in the same table row, combine it into the same title.
- If the page is one part of a multi-page TOC, extract only the rows visible on this page.
- Include courtFee and sheet count only if visible; otherwise keep them empty.
- Return [] if the page is not actually a TOC.

Return only valid JSON:
[
  {{
    "serialNo": "1",
    "title": "exact row description from the page",
    "pageFrom": 1,
    "pageTo": 4,
    "sheetCount": "",
    "courtFee": "",
    "source": "toc-image"
  }}
]"""
            try:
                raw = call_local_vision(image_b64, "image/jpeg", prompt, max_tokens=2600)
            except HTTPException as exc:
                log.warning("Skipping TOC image parsing for page %s after local vision failure: %s", page_num, exc.detail)
                continue

            parsed = extract_json_list(raw)
            if parsed:
                for row in parsed:
                    if isinstance(row, dict):
                        row.setdefault("source", "toc-image")
                toc_items.extend(parsed)
            elif raw.strip():
                log.info("TOC image response for page %s was not usable JSON (%s chars)", page_num, len(raw))
    finally:
        doc.close()

    return toc_items


def call_local_text(messages: list[dict], max_tokens: int = 2000, temperature: float = 0.1) -> str:
    """Call the configured local chat model for text-only reasoning tasks."""
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=LOCAL_LLM_TIMEOUT, write=LOCAL_LLM_TIMEOUT, pool=LOCAL_LLM_TIMEOUT)) as client:
            resp = client.post(
                f"{LOCAL_LLM_BASE_URL}/api/chat",
                json={
                    "model": LOCAL_TEXT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        log.exception("Local text model request failed")
        raise HTTPException(503, f"Local text model request failed: {exc}")

    data = resp.json()
    return ((data.get("message") or {}).get("content") or "").strip()


def call_local_vision(image_b64: str, media_type: str, prompt: str, max_tokens: int = 2000) -> str:
    """Call the configured local vision model with a single page image."""
    del media_type
    try:
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=LOCAL_LLM_TIMEOUT, write=LOCAL_LLM_TIMEOUT, pool=LOCAL_LLM_TIMEOUT)) as client:
            resp = client.post(
                f"{LOCAL_LLM_BASE_URL}/api/chat",
                json={
                    "model": LOCAL_VISION_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max_tokens,
                    },
                },
            )
            resp.raise_for_status()
    except Exception as exc:
        log.exception("Local vision model request failed")
        raise HTTPException(503, f"Local vision model request failed: {exc}")

    data = resp.json()
    return ((data.get("message") or {}).get("content") or "").strip()


def _strip_markdown_fences(text: str) -> str:
    return re.sub(r"```json|```", "", text or "").strip()


def _iter_json_candidates(text: str):
    text = _strip_markdown_fences(text)
    if not text:
        return

    yield text

    for opener, closer in [("[", "]"), ("{", "}")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    yield text[start:idx + 1]
                    break


def safe_json(text: str):
    """Parse JSON from model output, stripping wrappers and trailing prose."""
    for candidate in _iter_json_candidates(text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def extract_json_list(text: str) -> list[dict]:
    parsed = safe_json(text)
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("items", "rows", "index", "data", "toc"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def coerce_page_number(value, fallback: int) -> int:
    """Best-effort page parsing so noisy model output does not crash indexing."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return fallback


def tokenize_for_search(text: str) -> list[str]:
    return [token for token in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if len(token) > 1]


def lexical_overlap_score(question: str, page_text: str) -> float:
    q_tokens = tokenize_for_search(question)
    if not q_tokens:
        return 0.0

    page_tokens = tokenize_for_search(page_text)
    if not page_tokens:
        return 0.0

    page_set = set(page_tokens)
    overlap = sum(1 for token in q_tokens if token in page_set)
    phrase_bonus = 2.5 if question.strip() and question.lower() in (page_text or "").lower() else 0.0
    density_bonus = overlap / max(len(set(q_tokens)), 1)
    return overlap + density_bonus + phrase_bonus


def normalize_index_items(items: list[dict], indexed_start: int, indexed_end: int, default_source: str) -> list[dict]:
    normalized = []
    for item in items:
        pf = coerce_page_number(item.get("pageFrom"), indexed_start)
        pt = coerce_page_number(item.get("pageTo"), pf)
        if pt < pf:
            pt = pf
        pf = max(indexed_start, min(pf, indexed_end))
        pt = max(pf, min(pt, indexed_end))
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        normalized.append({
            "title": title,
            "displayTitle": str(item.get("displayTitle") or item.get("originalTitle") or title).strip(),
            "originalTitle": str(item.get("originalTitle") or title).strip(),
            "pageFrom": pf,
            "pageTo": pt,
            "source": item.get("source", default_source),
            "serialNo": str(item.get("serialNo", "")),
            "courtFee": str(item.get("courtFee", "")),
        })

    normalized.sort(key=lambda x: (x["pageFrom"], x["pageTo"], x["title"]))

    deduped = []
    seen = set()
    for item in normalized:
        key = (item["title"], item["pageFrom"], item["pageTo"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_toc_ranges_from_items(items: list[dict], indexed_start: int, range_end: int, default_source: str) -> list[dict]:
    normalized = normalize_index_items(items, indexed_start, range_end, default_source)
    if not normalized:
        return []

    ranged = []
    for idx, item in enumerate(normalized):
        current = dict(item)
        next_start = normalized[idx + 1]["pageFrom"] if idx + 1 < len(normalized) else None
        if next_start is not None and next_start > current["pageFrom"]:
            current["pageTo"] = max(current["pageFrom"], next_start - 1)
        else:
            current["pageTo"] = max(current["pageFrom"], min(current["pageTo"], range_end))
        current["pageTo"] = min(current["pageTo"], range_end)
        ranged.append(current)
    return ranged


def analyze_toc_page_features(text: str, toc_markers: list[str]) -> dict:
    content = text or ""
    lower = content.lower()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    header_hits = sum(
        1
        for pattern in TOC_TABLE_HEADER_PATTERNS
        if re.search(pattern, lower, flags=re.IGNORECASE)
    )
    row_like_lines = [
        line for line in lines
        if re.search(r"\d", line) and re.search(r".+?\d+\s*$", line)
    ]
    short_row_like_lines = [line for line in row_like_lines if len(line) <= 140]
    numbered_row_lines = [
        line for line in lines
        if re.match(r"^\s*[\[(]?\d{1,3}[\])\.\-]?\s+\S+", line)
    ]
    page_range_hits = len(re.findall(r"\b\d+\s*[\-?]\s*\d+\b", content))
    dotted_leaders = len(re.findall(r"\.{3,}", content))
    marker_hits = sum(1 for marker in toc_markers if marker in lower)
    line_hits = len(re.findall(r"\n\s*\d+\s+.+?\d+\s*$", content, flags=re.MULTILINE))
    return {
        "marker_hits": marker_hits,
        "header_hits": header_hits,
        "dotted_leaders": dotted_leaders,
        "line_hits": line_hits,
        "row_like_lines": len(row_like_lines),
        "short_row_like_lines": len(short_row_like_lines),
        "numbered_row_lines": len(numbered_row_lines),
        "page_range_hits": page_range_hits,
        "line_count": len(lines),
    }


def is_strong_toc_candidate(features: dict) -> bool:
    return (
        features["marker_hits"] >= 1
        or features["header_hits"] >= 2
        or features["line_hits"] >= 2
        or (features["header_hits"] >= 1 and features["numbered_row_lines"] >= 2)
        or (features["page_range_hits"] >= 2 and features["numbered_row_lines"] >= 2)
        or features["dotted_leaders"] >= 2
        or features["short_row_like_lines"] >= 4
    )


def is_toc_continuation_page(features: dict) -> bool:
    return (
        features["short_row_like_lines"] >= 3
        or features["numbered_row_lines"] >= 3
        or (
            features["row_like_lines"] >= 2
            and features["dotted_leaders"] >= 1
        )
        or (
            features["header_hits"] >= 1
            and features["page_range_hits"] >= 1
        )
        or (
            features["row_like_lines"] >= 4
            and features["line_count"] <= 35
        )
    )


def collect_toc_candidate_pages(all_pages: list[dict], toc_markers: list[str], max_pages: int = 6) -> list[dict]:
    candidates = []
    idx = 0
    while idx < len(all_pages) and len(candidates) < max_pages:
        page = all_pages[idx]
        features = analyze_toc_page_features(page["text"], toc_markers)
        if not is_strong_toc_candidate(features):
            idx += 1
            continue

        candidates.append(page)
        previous_page_num = page["page_num"]
        idx += 1

        while idx < len(all_pages) and len(candidates) < max_pages:
            next_page = all_pages[idx]
            if next_page["page_num"] != previous_page_num + 1:
                break
            next_features = analyze_toc_page_features(next_page["text"], toc_markers)
            if not (is_strong_toc_candidate(next_features) or is_toc_continuation_page(next_features)):
                break
            candidates.append(next_page)
            previous_page_num = next_page["page_num"]
            idx += 1
    return candidates


def collect_toc_fallback_pages(all_pages: list[dict], toc_markers: list[str], max_pages: int = 6) -> list[dict]:
    scored = []
    for page in all_pages[:max(max_pages, min(len(all_pages), 12))]:
        features = analyze_toc_page_features(page["text"], toc_markers)
        score = (
            (features["marker_hits"] * 5)
            + (features["header_hits"] * 4)
            + (features["numbered_row_lines"] * 2)
            + features["page_range_hits"]
            + features["short_row_like_lines"]
        )
        if score <= 0:
            continue
        scored.append((score, page["page_num"], page))

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = []
    seen = set()
    for _, _, page in scored:
        if page["page_num"] in seen:
            continue
        chosen.append(page)
        seen.add(page["page_num"])
        if len(chosen) >= max_pages:
            break
    return sorted(chosen, key=lambda item: item["page_num"])


def combine_toc_items(*groups: list[dict]) -> list[dict]:
    combined = []
    for group in groups:
        if isinstance(group, list):
            combined.extend(item for item in group if isinstance(item, dict))
    return combined


def extract_pages_from_document(
    doc: fitz.Document,
    page_numbers: list[int],
    total_pages: int,
    dpi: int = 250,
    timing_collector: Optional[PdfTimingCollector] = None,
) -> tuple[list[dict], dict]:
    pages_data = []
    ocr_count = 0
    vision_ocr_count = 0
    handwriting_count = 0

    for page_num in page_numbers:
        page = doc[page_num - 1]
        page_data = extract_page_content(page, page_num, dpi=dpi, timing_collector=timing_collector)
        text_value = page_data["text"] or f"[Page {page_num} - no readable text detected]"
        if page_data["used_ocr"]:
            ocr_count += 1
        if page_data["vision_used"]:
            vision_ocr_count += 1
        if page_data["handwriting_suspected"]:
            handwriting_count += 1

        pages_data.append({
            "page_num": page_num,
            "text": text_value,
            "used_ocr": page_data["used_ocr"],
            "vision_used": page_data["vision_used"],
            "handwriting_suspected": page_data["handwriting_suspected"],
            "extraction_method": page_data["extraction_method"],
        })
        log.info(
            "  Page %s/%s - %s%s - %s chars",
            page_num,
            total_pages,
            page_data["extraction_method"],
            " (handwriting assist)" if page_data["vision_used"] else "",
            len(text_value),
        )

    return pages_data, {
        "ocr_pages": ocr_count,
        "vision_ocr_pages": vision_ocr_count,
        "handwriting_suspected_pages": handwriting_count,
        "digital_pages": len(page_numbers) - ocr_count,
    }


def upsert_collection_pages(
    pdf_id: str,
    filename: str,
    pages_data: list[dict],
    reset: bool = False,
    timing_collector: Optional[PdfTimingCollector] = None,
):
    if reset:
        try:
            chroma_client.delete_collection(f"pdf_{pdf_id}")
        except Exception:
            pass
    collection = get_or_create_collection(pdf_id)
    if not pages_data:
        return collection

    batch_size = 50
    vectorization_scope = timing_collector.stage("total vectorization", "total_vectorization_time") if timing_collector else nullcontext()
    with vectorization_scope:
        for i in range(0, len(pages_data), batch_size):
            batch = pages_data[i:i + batch_size]

            chunk_started = time.perf_counter()
            ids = [f"{pdf_id}_p{page['page_num']}" for page in batch]
            documents = [page["text"] for page in batch]
            metadatas = [{
                "page_num": page["page_num"],
                "used_ocr": page["used_ocr"],
                "vision_used": page["vision_used"],
                "handwriting_suspected": page["handwriting_suspected"],
                "extraction_method": page["extraction_method"],
                "filename": filename,
            } for page in batch]
            if timing_collector:
                timing_collector.add_duration("chunking_time", time.perf_counter() - chunk_started)

            embedding_started = time.perf_counter()
            embeddings = embed_texts(documents)
            if timing_collector:
                timing_collector.add_duration("embedding_time", time.perf_counter() - embedding_started)

            db_insert_started = time.perf_counter()
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
            if timing_collector:
                timing_collector.add_duration("vector_db_insert_time", time.perf_counter() - db_insert_started)
    return collection

def load_collection_pages(pdf_id: str) -> list[dict]:
    try:
        collection = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        return []

    result = collection.get(include=["documents", "metadatas"])
    pages = []
    for doc_text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        pages.append({
            "page_num": meta["page_num"],
            "text": doc_text,
            "used_ocr": bool(meta.get("used_ocr")),
            "vision_used": bool(meta.get("vision_used")),
            "handwriting_suspected": bool(meta.get("handwriting_suspected")),
            "extraction_method": meta.get("extraction_method", "unknown"),
            "stage": "legacy",
        })
    pages.sort(key=lambda item: item["page_num"])
    return pages


def load_index_pages(pdf_id: str) -> list[dict]:
    cached_pages = get_cached_pages(pdf_id)
    if cached_pages:
        return cached_pages
    return load_collection_pages(pdf_id)


def build_segment_preview(all_pages: list[dict], page_from: int, page_to: int, max_chars: int = 1400) -> str:
    parts = []
    char_count = 0
    for page in all_pages:
        if page["page_num"] < page_from or page["page_num"] > page_to:
            continue
        snippet = (page["text"] or "").strip()
        if not snippet:
            continue
        snippet = snippet[:600]
        parts.append(f"Page {page['page_num']}: {snippet}")
        char_count += len(snippet)
        if char_count >= max_chars:
            break
    return "\n".join(parts)


def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text or ""))


def normalize_label(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/,()\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_structural_toc_title(raw_title: str) -> bool:
    normalized = normalize_label(raw_title)
    if not normalized:
        return False
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in STRUCTURAL_TOC_PATTERNS)


def should_preserve_original_title(item: dict, raw_title: str, chosen_title: str, scored: list[tuple[float, str]]) -> bool:
    normalized_raw = normalize_label(raw_title)
    normalized_chosen = normalize_label(chosen_title)
    source = item.get("source", "")

    if not normalized_raw or not normalized_chosen:
        return False

    if is_structural_toc_title(raw_title):
        return True

    blocked = set()
    for key, blocked_titles in NEGATIVE_TOC_MAPPINGS.items():
        if key in normalized_raw:
            blocked.update(blocked_titles)
    if chosen_title in blocked:
        return True

    score_map = {name: score for score, name in scored}
    top_score = score_map.get(chosen_title, 0.0)
    second_score = max((score for score, name in scored if name != chosen_title), default=-999.0)
    if source in LOW_CONFIDENCE_TOC_SOURCES and (top_score < 6.8 or (top_score - second_score) < 1.7):
        return True

    if source in LOW_CONFIDENCE_TOC_SOURCES and normalized_chosen not in normalized_raw and lexical_overlap_score(chosen_title, raw_title) < 1.0:
        return True

    return False


def direct_parent_match(raw_title: str, preview: str) -> Optional[str]:
    combined = f"{raw_title}\n{preview}".lower()
    normalized_combined = normalize_label(combined)

    alias_rules = [
        (r"\b(table of contents|index)\b|सूची|विषय सूची|अनुक्रमणिका|क्रमानुसार", "Index"),
        (r"vakalat|वकालतनामा", "Vakalat Nama"),
        (r"written statement|लिखित", "Written Statement"),
        (r"\brejoinder\b", "Rejoinder"),
        (r"\breply\b|जवाब", "Reply"),
        (r"\breplication\b", "Replication"),
        (r"affidavit|शपथ", "Affidavit"),
        (r"power of attorney", "Power of Attorney"),
        (r"memo of parties", "Memo of Parties"),
        (r"list of dates|dates and events", "List of Dates & Events"),
        (r"brief synopsis|synopsis", "Brief Synopsis"),
        (r"annexure|अनुलग्न|संलग्न", "Annexure"),
        (r"impugned order|आदेश", "Impugned Order"),
        (r"application|प्रार्थना पत्र|अर्जी", "Application"),
        (r"court fee|stamp paper", "e-Court Fee/Stamp Paper"),
        (r"final order|अंतिम आदेश", "FINAL ORDER"),
        (r"office note", "Office Note"),
        (r"administrative order", "Administrative Orders"),
        (r"notice|सूचना", "Notices"),
        (r"letter", "Letter"),
        (r"paper book", "Paper Book"),
        (r"report|प्रतिवेदन", "Reports"),
        (r"identity proof|पहचान", "Identity Proof"),
        (r"process fee", "Process Fee"),
        (r"urgent form|urgency", "Urgent Form"),
    ]
    for pattern, target in alias_rules:
        if re.search(pattern, combined, flags=re.IGNORECASE) and target in PARENT_DOCUMENT_NAMES:
            return target

    exact_map = {normalize_label(name): name for name in PARENT_DOCUMENT_NAMES}
    for normalized_name, original_name in exact_map.items():
        if normalized_name and normalized_name in normalized_combined:
            return original_name
    return None


def score_parent_documents(raw_title: str, preview: str) -> list[tuple[float, str]]:
    if not PARENT_DOCUMENT_NAMES:
        return []

    segment_text = f"{raw_title}\n{preview}".strip()
    segment_vec = embed_texts([segment_text or raw_title or "document"])[0]
    parent_vecs = get_parent_document_embeddings()

    scored = []
    raw_lower = (raw_title or "").lower()
    for name, name_vec in zip(PARENT_DOCUMENT_NAMES, parent_vecs):
        normalized_name = normalize_label(name)
        lexical = lexical_overlap_score(name, segment_text)
        exact = 4.0 if normalized_name and normalized_name in normalize_label(raw_title) else 0.0
        preview_hit = 1.5 if normalized_name and normalized_name in normalize_label(preview) else 0.0
        semantic = sum(a * b for a, b in zip(segment_vec, name_vec))
        generic_penalty = -3.0 if name.lower() in GENERIC_PARENT_NAMES else 0.0
        score = (semantic * 2.8) + (lexical * 1.8) + exact + preview_hit + generic_penalty
        if name.lower() in raw_lower and name.lower() not in GENERIC_PARENT_NAMES:
            score += 2.0
        scored.append((score, name))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def get_parent_document_embeddings():
    global PARENT_DOCUMENT_EMBEDDINGS
    if PARENT_DOCUMENT_EMBEDDINGS is None and PARENT_DOCUMENT_NAMES:
        PARENT_DOCUMENT_EMBEDDINGS = embed_texts(PARENT_DOCUMENT_NAMES)
    return PARENT_DOCUMENT_EMBEDDINGS or []


def shortlist_parent_documents(raw_title: str, preview: str, top_n: int = 8) -> list[str]:
    direct = direct_parent_match(raw_title, preview)
    if direct:
        return [direct]

    scored = score_parent_documents(raw_title, preview)
    if not scored:
        return []

    preferred = [name for _, name in scored if name.lower() not in GENERIC_PARENT_NAMES]
    generic = [name for _, name in scored if name.lower() in GENERIC_PARENT_NAMES]
    if contains_devanagari(f"{raw_title}\n{preview}") and len(preferred) < 12:
        top_n = max(top_n, 12)
    picked = preferred[:top_n]
    if generic:
        picked.extend(generic[:1])
    return picked


def choose_parent_document(raw_title: str, preview: str, candidates: list[str], scored: list[tuple[float, str]]) -> str:
    direct = direct_parent_match(raw_title, preview)
    if direct:
        return direct

    if not candidates:
        preferred = next((name for name in PARENT_DOCUMENT_NAMES if name.lower() not in GENERIC_PARENT_NAMES), None)
        return preferred or (PARENT_DOCUMENT_NAMES[0] if PARENT_DOCUMENT_NAMES else "Other")

    if len(candidates) == 1:
        return candidates[0]

    score_map = {name: score for score, name in scored}
    top_score = score_map.get(candidates[0], 0.0)
    second_score = score_map.get(candidates[1], 0.0) if len(candidates) > 1 else -999.0
    if top_score >= 6.5 and (top_score - second_score) >= 1.5:
        return candidates[0]

    candidate_lines = "\n".join(f"- {name}" for name in candidates)
    prompt = f"""Choose the best parent document type for this indexed PDF range.

You must choose exactly one item from this candidate list:
{candidate_lines}

Raw title:
{raw_title}

Preview text:
{preview[:2000]}

Important rules:
- Prefer the most specific legal filing/document type from the candidate list.
- The raw title may be in Hindi, English, or mixed OCR.
- Use the preview pages to map Hindi/vernacular titles to the closest parent document field.
- Choose "Other" or "Others" only if nothing else genuinely fits.

Return only JSON:
{{"title": "one exact candidate name"}}"""

    raw = call_local_text(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0,
    )
    parsed = safe_json(raw)
    title = str((parsed or {}).get("title", "")).strip() if isinstance(parsed, dict) else ""
    return title if title in candidates else candidates[0]


def smooth_generic_ranges(items: list[dict]) -> list[dict]:
    if not items:
        return items

    smoothed = [dict(item) for item in items]
    for idx, item in enumerate(smoothed):
        if item.get("title", "").lower() not in GENERIC_PARENT_NAMES:
            continue
        prev_item = smoothed[idx - 1] if idx > 0 else None
        next_item = smoothed[idx + 1] if idx + 1 < len(smoothed) else None
        prev_title = (prev_item or {}).get("title", "")
        next_title = (next_item or {}).get("title", "")
        if prev_item and next_item and prev_title == next_title and prev_title.lower() not in GENERIC_PARENT_NAMES:
            item["title"] = prev_title
        elif prev_item and prev_title.lower() not in GENERIC_PARENT_NAMES and item.get("source") == "gap":
            item["title"] = prev_title
        elif next_item and next_title.lower() not in GENERIC_PARENT_NAMES and item.get("source") == "gap":
            item["title"] = next_title
    return smoothed


def merge_adjacent_ranges(items: list[dict]) -> list[dict]:
    if not items:
        return items

    merged = [dict(items[0])]
    for item in items[1:]:
        prev = merged[-1]
        if (
            item.get("title") == prev.get("title")
            and item.get("pageFrom") == prev.get("pageTo", 0) + 1
        ):
            prev["pageTo"] = item["pageTo"]
            prev["source"] = prev.get("source") if prev.get("source") != "gap" else item.get("source", prev.get("source"))
            if not prev.get("serialNo"):
                prev["serialNo"] = item.get("serialNo", "")
            if not prev.get("courtFee"):
                prev["courtFee"] = item.get("courtFee", "")
            continue
        merged.append(dict(item))
    return merged


def classify_index_to_parent_documents(index_items: list[dict], all_pages: list[dict]) -> list[dict]:
    """Restrict final index titles to the fixed parent document catalog only."""
    if not index_items or not PARENT_DOCUMENT_NAMES:
        return index_items

    classified = []
    for item in index_items:
        raw_title = item.get("title", "")
        preview = build_segment_preview(all_pages, item.get("pageFrom", 1), item.get("pageTo", 1))
        scored = score_parent_documents(raw_title, preview)
        candidates = shortlist_parent_documents(raw_title, preview)
        chosen_title = choose_parent_document(raw_title, preview, candidates, scored)
        title = raw_title if should_preserve_original_title(item, raw_title, chosen_title, scored) else chosen_title
        classified.append({
            **item,
            "title": title,
            "displayTitle": item.get("displayTitle") or raw_title or title,
            "originalTitle": item.get("originalTitle") or raw_title or title,
        })
    return merge_adjacent_ranges(smooth_generic_ranges(classified))


# ═════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    pdf_id: str
    question: str
    top_k: int = 8
    current_page: Optional[int] = None

class IndexRequest(BaseModel):
    pdf_id: str

class TextTransformRequest(BaseModel):
    text: str
    action: str


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

def cnr_number_from_filename(filename: str) -> str:
    return Path(filename or "").stem.strip()


def build_existing_pdf_payload(pdf_id: str, filename_override: str = "") -> Optional[dict]:
    record = get_pdf_record(pdf_id)
    pdf_path = stored_pdf_path(pdf_id)
    if not record or not pdf_path.exists():
        return None
    saved_index = get_saved_index(pdf_id)
    return {
        "pdf_id": pdf_id,
        "cnr_number": record.get("cnr_number") or cnr_number_from_filename(filename_override or record.get("filename") or ""),
        "total_pages": record.get("total_pages", 0),
        "indexed_pages": record.get("indexed_pages", 0),
        "indexed_page_start": record.get("selected_start_page", 1),
        "indexed_page_end": record.get("selected_end_page", 1),
        "ocr_pages": 0,
        "vision_ocr_pages": 0,
        "handwriting_suspected_pages": 0,
        "digital_pages": 0,
        "status": record.get("status", "index_ready"),
        "retrieval_status": record.get("retrieval_status", "pending_deferred_ingestion"),
        "pending_pages": record.get("pending_pages", 0),
        "chat_ready": bool(record.get("chat_ready")),
        "filename": record.get("filename") or filename_override or f"{pdf_id}.pdf",
        "index": saved_index,
        "index_entries": len(saved_index),
        "index_ready": bool(record.get("index_ready")),
        "index_source": record.get("index_source", "saved"),
        "skipped_duplicate": True,
        "message": "This PDF is already saved in the backend. Skipping duplicate upload.",
    }


def run_stage_one_ingest(pdf_bytes: bytes, filename: str, start_page: int = 1, end_page: Optional[int] = None) -> dict:
    pdf_id = pdf_id_from_bytes(pdf_bytes)
    pdf_path = stored_pdf_path(pdf_id)
    pdf_path.write_bytes(pdf_bytes)
    log.info("Stage 1 ingest for PDF: %s  id=%s", filename, pdf_id)
    timing_collector = PdfTimingCollector(pdf_id, filename)

    with timing_collector.stage("file open", "file_open"):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    if total_pages < 1:
        doc.close()
        raise HTTPException(400, "PDF has no pages")

    start_page = max(1, min(start_page, total_pages))
    default_end_page = min(total_pages, start_page + 9)
    end_page = default_end_page if end_page is None else min(end_page, total_pages)
    if start_page > end_page:
        doc.close()
        raise HTTPException(400, "Start page must be less than or equal to end page")

    selected_page_numbers = list(range(start_page, end_page + 1))
    try:
        with timing_collector.stage("first 10-page extraction", "first_10_page_extraction"):
            pages_data, stats = extract_pages_from_document(doc, selected_page_numbers, total_pages, dpi=250, timing_collector=timing_collector)
    finally:
        doc.close()

    replace_extracted_pages(pdf_id, pages_data, stage="fast_index")
    upsert_collection_pages(pdf_id, filename, pages_data, reset=True, timing_collector=timing_collector)

    indexed_pages = len(pages_data)
    pending_pages = max(total_pages - indexed_pages, 0)
    cnr_number = cnr_number_from_filename(filename)
    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=filename,
        cnr_number=cnr_number,
        file_size_bytes=len(pdf_bytes),
        total_pages=total_pages,
        selected_start_page=start_page,
        selected_end_page=end_page,
        indexed_pages=indexed_pages,
        status="toc_scanned",
        retrieval_status="pending_deferred_ingestion" if pending_pages else "vectorized",
        index_ready=False,
        chat_ready=not bool(pending_pages),
        pending_pages=pending_pages,
        index_source="",
        queue_bucket="deferred" if pending_pages else "library",
        deferred_decision="pending" if pending_pages else "completed",
        last_error="",
    )

    timing_collector.log_summary("stage_1_ingest")

    return {
        "pdf_id": pdf_id,
        "cnr_number": cnr_number,
        "total_pages": total_pages,
        "indexed_pages": indexed_pages,
        "indexed_page_start": start_page,
        "indexed_page_end": end_page,
        "ocr_pages": stats["ocr_pages"],
        "vision_ocr_pages": stats["vision_ocr_pages"],
        "handwriting_suspected_pages": stats["handwriting_suspected_pages"],
        "digital_pages": stats["digital_pages"],
        "status": "toc_scanned",
        "retrieval_status": "pending_deferred_ingestion" if pending_pages else "vectorized",
        "pending_pages": pending_pages,
        "chat_ready": not bool(pending_pages),
        "filename": filename,
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"vision": LOCAL_VISION_MODEL, "text": LOCAL_TEXT_MODEL},
        "llm_base_url": LOCAL_LLM_BASE_URL,
        "embedding_ready": embedder is not None,
        "handwritten_hindi_assist": ENABLE_HANDWRITTEN_HINDI_ASSIST and bool(LOCAL_VISION_MODEL),
        "workflow_storage_backend": WORKFLOW_STORAGE_BACKEND,
        "workflow_storage_target": WORKFLOW_STORAGE_TARGET,
    }


# ── 1. INGEST PDF ─────────────────────────────────────────────────────────────
async def build_stage_one_payload_async(pdf_bytes: bytes, filename: str, start_page: int = 1, end_page: Optional[int] = None) -> dict:
    ingest_resp = run_stage_one_ingest(pdf_bytes, filename, start_page=start_page, end_page=end_page)
    index_resp = await generate_index(IndexRequest(pdf_id=ingest_resp["pdf_id"]))
    return {
        **ingest_resp,
        "index": index_resp.get("index", []),
        "index_source": index_resp.get("index_source", ingest_resp.get("status")),
        "status": index_resp.get("status", ingest_resp.get("status")),
        "retrieval_status": index_resp.get("retrieval_status", ingest_resp.get("retrieval_status")),
        "pending_pages": index_resp.get("pending_pages", ingest_resp.get("pending_pages")),
        "chat_ready": index_resp.get("chat_ready", ingest_resp.get("chat_ready")),
        "indexed_page_start": index_resp.get("indexed_page_start", ingest_resp.get("indexed_page_start")),
        "indexed_page_end": index_resp.get("indexed_page_end", ingest_resp.get("indexed_page_end")),
        "indexed_pages": index_resp.get("indexed_pages", ingest_resp.get("indexed_pages")),
    }


def build_stage_one_payload(pdf_bytes: bytes, filename: str, start_page: int = 1, end_page: Optional[int] = None) -> dict:
    return asyncio.run(build_stage_one_payload_async(pdf_bytes, filename, start_page=start_page, end_page=end_page))


def run_stage_one_index_worker(pdf_bytes: bytes, filename: str, start_page: int, end_page: Optional[int], known_pdf_id: str = ""):
    pdf_id = known_pdf_id or pdf_id_from_bytes(pdf_bytes)
    index_runner_status.update({
        "running": True,
        "current_pdf_id": pdf_id,
        "current_filename": filename,
        "last_error": "",
        "heartbeat_ts": time.time(),
        "started_at": time.time(),
        "finished_pdf_id": "",
        "finished_filename": "",
        "status": "running",
    })
    try:
        build_stage_one_payload(pdf_bytes, filename, start_page=start_page, end_page=end_page)
        index_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
            "finished_pdf_id": pdf_id,
            "finished_filename": filename,
            "status": "completed",
        })
    except Exception as exc:
        log.exception("Stage 1 background indexing failed for %s", filename)
        update_pdf_record(pdf_id, status="failed", last_error=str(exc))
        index_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": str(exc),
            "heartbeat_ts": time.time(),
            "finished_pdf_id": pdf_id,
            "finished_filename": filename,
            "status": "failed",
        })


@app.post("/api/index-runner/upload")
async def start_background_index_upload(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    with index_runner_lock:
        if index_runner_status.get("running"):
            return {
                "started": False,
                "runner": dict(index_runner_status),
                "message": "Background indexing is already running.",
            }

        pdf_bytes = await file.read()
        pdf_id = pdf_id_from_bytes(pdf_bytes)
        existing = build_existing_pdf_payload(pdf_id, file.filename)
        if existing:
            return {
                "started": False,
                "pdf_id": pdf_id,
                "filename": existing.get("filename") or file.filename,
                "runner": dict(index_runner_status),
                "message": existing.get("message"),
                "skipped_duplicate": True,
                "existing": existing,
            }
        worker = Thread(
            target=run_stage_one_index_worker,
            args=(pdf_bytes, file.filename, start_page, end_page, pdf_id),
            daemon=True,
        )
        worker.start()

    return {
        "started": True,
        "pdf_id": pdf_id,
        "filename": file.filename,
        "runner": dict(index_runner_status),
    }


@app.post("/api/index-runner/saved/{pdf_id}")
async def start_background_index_saved(
    pdf_id: str,
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    record = get_pdf_record(pdf_id)
    pdf_path = stored_pdf_path(pdf_id)
    if not record or not pdf_path.exists():
        raise HTTPException(404, f"PDF {pdf_id} not found")

    with index_runner_lock:
        if index_runner_status.get("running"):
            return {
                "started": False,
                "runner": dict(index_runner_status),
                "message": "Background indexing is already running.",
            }

        pdf_bytes = pdf_path.read_bytes()
        worker = Thread(
            target=run_stage_one_index_worker,
            args=(pdf_bytes, record.get("filename") or f"{pdf_id}.pdf", start_page, end_page, pdf_id),
            daemon=True,
        )
        worker.start()

    return {
        "started": True,
        "pdf_id": pdf_id,
        "filename": record.get("filename") or f"{pdf_id}.pdf",
        "runner": dict(index_runner_status),
    }


@app.post("/api/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    """
    Stage 1 fast indexing pipeline.
    Save the PDF, scan only the selected window, cache extracted pages,
    vectorize that subset, and mark the rest for deferred ingestion.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_bytes = await file.read()
    return run_stage_one_ingest(pdf_bytes, file.filename, start_page=start_page, end_page=end_page)


@app.post("/api/pdfs/{pdf_id}/reindex-saved")
async def reindex_saved_pdf(
    pdf_id: str,
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    """
    Re-run Stage 1 indexing for an already-saved PDF without requiring the browser
    to upload the file again.
    """
    record = get_pdf_record(pdf_id)
    pdf_path = stored_pdf_path(pdf_id)
    if not record or not pdf_path.exists():
        raise HTTPException(404, f"PDF {pdf_id} not found")

    pdf_bytes = pdf_path.read_bytes()
    return await build_stage_one_payload_async(
        pdf_bytes,
        record.get("filename") or f"{pdf_id}.pdf",
        start_page=start_page,
        end_page=end_page,
    )


# -- 2. QUERY / CHATBOT ────────────────────────────────────────────────────────
@app.post("/api/query")
async def query_pdf(req: QueryRequest):
    """
    Answer a natural language question about the PDF using RAG.
    Chat/search becomes fully available only after deferred ingestion completes.
    """
    record = get_pdf_record(req.pdf_id)
    if record and not record.get("chat_ready"):
        raise HTTPException(409, "Chat/search will be available after deferred ingestion finishes for this PDF.")

    try:
        collection = chroma_client.get_collection(f"pdf_{req.pdf_id}")
    except Exception:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_rows = []
    all_results = collection.get(include=["documents", "metadatas", "embeddings"])
    for doc_text, meta, emb in zip(all_results["documents"], all_results["metadatas"], all_results["embeddings"]):
        all_rows.append({
            "page_num": meta["page_num"],
            "text": doc_text,
            "embedding": emb,
        })

    q_vec = embed_texts([req.question])[0]
    q_tokens = tokenize_for_search(req.question)
    scored_rows = []

    for row in all_rows:
        lexical = lexical_overlap_score(req.question, row["text"])
        semantic = sum(a * b for a, b in zip(q_vec, row["embedding"]))
        nearby = 0.0
        if req.current_page is not None and abs(row["page_num"] - req.current_page) <= 2:
            nearby = 1.5
        if req.current_page is not None and row["page_num"] == req.current_page:
            nearby = 3.0
        token_presence = 0.0
        if q_tokens:
            page_lower = (row["text"] or "").lower()
            token_presence = sum(1 for token in q_tokens if token in page_lower) / len(q_tokens)

        score = (semantic * 2.0) + lexical + nearby + token_presence
        scored_rows.append({**row, "score": score})

    scored_rows.sort(key=lambda row: (row["score"], row["page_num"]), reverse=True)
    top_rows = [row for row in scored_rows[: max(3, min(req.top_k, len(scored_rows)))] if row["score"] > 0]
    if not top_rows:
        top_rows = scored_rows[: min(req.top_k, len(scored_rows))]

    context_parts = []
    page_refs = []
    for row in top_rows:
        page_refs.append(row["page_num"])
        context_parts.append(f"--- Page {row['page_num']} ---\n{row['text'][:1400]}")

    context = "\n\n".join(context_parts)
    system_prompt = """You are an expert assistant for Indian court documents.
The documents may contain Hindi (Devanagari script), English, or mixed text.
Answer questions accurately based ONLY on the provided page content.
Always mention which page number your answer comes from.
If the answer is not in the provided pages, say so clearly.
For Hindi text, keep it in Devanagari script in your answer."""

    user_prompt = f"""Question: {req.question}

Relevant pages from the document:
{context}

Please answer the question based on the above pages. Mention page numbers.
If the current page appears relevant, prioritize that page and its nearby pages."""

    answer = call_local_text(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    return {
        "answer": answer,
        "page_refs": sorted(list(set(page_refs))),
        "chunks_used": len(top_rows),
    }


# -- 3. GENERATE INDEX ─────────────────────────────────────────────────────────
@app.post("/api/generate-index")
async def generate_index(req: IndexRequest):
    """
    Generate the structured document index from cached Stage 1 pages first.
    If a TOC is detected, ranges are expanded deterministically across the whole PDF.
    """
    record = get_pdf_record(req.pdf_id)
    timing_collector = PdfTimingCollector(req.pdf_id, (record or {}).get("filename", ""))
    all_pages = load_index_pages(req.pdf_id)
    if not all_pages:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_pages.sort(key=lambda item: item["page_num"])
    indexed_start = all_pages[0]["page_num"]
    indexed_end = all_pages[-1]["page_num"]
    total_pages = record["total_pages"] if record else indexed_end
    total_chunks = len(all_pages)

    toc_markers = [
        "table of contents",
        "contents",
        "index",
        "????",
        "???? ????",
        "???????????",
        "???????",
        "?????",
        "pages",
        "sheet",
    ]
    def run_toc_extraction(candidate_pages: list[dict]) -> tuple[list[int], list[dict], list[dict]]:
        toc_pages_text = ""
        toc_page_nums = []
        for page in candidate_pages:
            toc_pages_text += f"\n--- Page {page['page_num']} ---\n{page['text']}\n"
            toc_page_nums.append(page["page_num"])

        toc_image_items_local = []
        toc_text_items_local = []
        if toc_pages_text:
            toc_image_items_local = extract_toc_from_page_images(stored_pdf_path(req.pdf_id), toc_page_nums)
            if toc_image_items_local:
                log.info("TOC extracted from page images: %s items from pages %s", len(toc_image_items_local), toc_page_nums)

            toc_prompt = f"""You are reading OCR/text extracted from Indian court document pages.
These pages may contain a real Table of Contents / Index / ???? / ???? ???? / ???????????.
The TOC may continue across 2 or more consecutive pages.

If the pages contain a table of contents, extract the rows exactly as written and build the final index from that TOC.
Use both English and Hindi entries if present.
Treat continuation pages as part of the same TOC when they keep the same row/table structure.

Rules:
- Preserve original title text exactly. Do not translate Hindi.
- Focus on tabular document rows, not party names or case-title headers above the table.
- Read serial numbers, page numbers, annexure values, and page ranges carefully.
- Hindi digits should be converted to Arabic numerals in pageFrom/pageTo.
- If a row has only one page number, set pageFrom and pageTo to the same page.
- If a row title wraps across multiple OCR lines, merge it into the same row title.
- Include every TOC row you can confidently read from all shown pages.
- If these pages are not actually a TOC, return [].

Extracted text from pages {toc_page_nums}:
{toc_pages_text[:9000]}

Return only valid JSON:
[
  {{
    "serialNo": "1",
    "title": "exact original text as found",
    "pageFrom": 1,
    "pageTo": 4,
    "courtFee": "",
    "source": "toc"
  }}
]"""

            toc_raw = call_local_text(
                messages=[{"role": "user", "content": toc_prompt}],
                max_tokens=3500,
            )
            toc_text_items_local = extract_json_list(toc_raw)
            if toc_text_items_local:
                log.info("TOC extracted from text: %s items from pages %s", len(toc_text_items_local), toc_page_nums)
            elif toc_raw.strip():
                log.info("TOC text response for pages %s was not usable JSON (%s chars)", toc_page_nums, len(toc_raw))

        return toc_page_nums, toc_image_items_local, toc_text_items_local

    index_items = []
    toc_source = ""
    toc_range_end = total_pages if record else indexed_end
    toc_page_nums = []
    toc_image_items = []
    toc_text_items = []

    with timing_collector.stage("TOC detection", "toc_detection"):
        toc_candidate_pages = collect_toc_candidate_pages(all_pages, toc_markers, max_pages=6)
        log.info("Primary TOC candidate pages for %s: %s", req.pdf_id, [page["page_num"] for page in toc_candidate_pages])

        if toc_candidate_pages:
            toc_page_nums, toc_image_items, toc_text_items = run_toc_extraction(toc_candidate_pages)

        combined_toc_items = combine_toc_items(toc_image_items, toc_text_items)
        if not combined_toc_items:
            fallback_toc_pages = collect_toc_fallback_pages(all_pages, toc_markers, max_pages=6)
            fallback_page_nums = [page["page_num"] for page in fallback_toc_pages]
            if fallback_page_nums and fallback_page_nums != toc_page_nums:
                log.info("Retrying TOC extraction with fallback pages for %s: %s", req.pdf_id, fallback_page_nums)
                toc_page_nums, toc_image_items, toc_text_items = run_toc_extraction(fallback_toc_pages)
                combined_toc_items = combine_toc_items(toc_image_items, toc_text_items)
        if combined_toc_items:
            index_items = build_toc_ranges_from_items(combined_toc_items, indexed_start, toc_range_end, "toc")
            toc_source = "toc-image" if toc_image_items and not toc_text_items else "toc"
            log.info("TOC merged across pages %s into %s unique rows", toc_page_nums, len(index_items))

    with timing_collector.stage("LLM indexing", "llm_indexing_time"):
        scan_chunk = 25
        auto_items = []
        uncovered_pages = all_pages if not index_items else []
        if not index_items:
            log.info("No TOC found. Scanning %s indexed pages for document boundaries", len(uncovered_pages))

        for i in range(0, len(uncovered_pages), scan_chunk):
            chunk_pages = uncovered_pages[i:i + scan_chunk]
            if not chunk_pages:
                continue

            page_texts = ""
            for pg in chunk_pages:
                preview = pg["text"][:300].replace("\n", " ").strip()
                page_texts += f"Page {pg['page_num']}: {preview}\n"

            from_page = chunk_pages[0]["page_num"]
            to_page = chunk_pages[-1]["page_num"]

            scan_prompt = f"""You are analyzing pages from an Indian court document.
Below is extracted text from pages {from_page} to {to_page}.
Each line shows the page number and a preview of its content.

Identify each distinct document or section that STARTS within these pages.
Look for: document titles, application headers, order headings, affidavit starts, annexure labels, etc.
Text may be in Hindi (Devanagari), English, or mixed - preserve it exactly.

Page content:
{page_texts}

Return ONLY a valid JSON array (empty [] if nothing found):
[
  {{
    "title": "exact title as it appears on the page - original language",
    "pageFrom": {from_page},
    "pageTo": {to_page},
    "source": "auto"
  }}
]"""

            raw = call_local_text(
                messages=[{"role": "user", "content": scan_prompt}],
                max_tokens=2000,
            )
            parsed = safe_json(raw)
            if isinstance(parsed, list):
                for item in parsed:
                    if item.get("title") and len(item["title"]) > 2:
                        auto_items.append({
                            **item,
                            "pageFrom": max(from_page, coerce_page_number(item.get("pageFrom"), from_page)),
                            "pageTo": min(to_page, coerce_page_number(item.get("pageTo"), to_page)),
                            "source": "auto",
                        })

        all_items = index_items + auto_items
        all_items.sort(key=lambda item: item.get("pageFrom", 0))

        final = []
        cursor = indexed_start
        final_range_end = total_pages if index_items else indexed_end
        for item in all_items:
            pf = coerce_page_number(item.get("pageFrom"), cursor)
            pt = coerce_page_number(item.get("pageTo"), pf)
            if pt < pf:
                pt = pf
            if pf > cursor:
                final.append({
                    "title": f"Pages {cursor}-{pf - 1}",
                    "displayTitle": f"Pages {cursor}-{pf - 1}",
                    "originalTitle": f"Pages {cursor}-{pf - 1}",
                    "pageFrom": cursor,
                    "pageTo": pf - 1,
                    "source": "gap",
                    "serialNo": "",
                    "courtFee": "",
                })
            if pf >= cursor:
                final.append({
                    "title": item.get("title", f"Pages {pf}-{pt}"),
                    "displayTitle": item.get("displayTitle") or item.get("originalTitle") or item.get("title", f"Pages {pf}-{pt}"),
                    "originalTitle": item.get("originalTitle") or item.get("title", f"Pages {pf}-{pt}"),
                    "pageFrom": pf,
                    "pageTo": pt,
                    "source": item.get("source", "auto"),
                    "serialNo": str(item.get("serialNo", "")),
                    "courtFee": str(item.get("courtFee", "")),
                })
                cursor = pt + 1

        if cursor <= final_range_end:
            final.append({
                "title": f"Pages {cursor}-{final_range_end}",
                "displayTitle": f"Pages {cursor}-{final_range_end}",
                "originalTitle": f"Pages {cursor}-{final_range_end}",
                "pageFrom": cursor,
                "pageTo": final_range_end,
                "source": "gap",
                "serialNo": "",
                "courtFee": "",
            })

        classified_final = classify_index_to_parent_documents(final, all_pages)
        index_source = toc_source or ("toc" if index_items else "auto")

    with timing_collector.stage("JSON generation", "json_generation_time"):
        save_index(req.pdf_id, classified_final)

        if record:
            update_pdf_record(
                req.pdf_id,
                status="index_ready",
                index_ready=True,
                index_source=index_source,
                queue_bucket="deferred" if record.get("pending_pages", 0) > 0 else "library",
                deferred_decision="pending" if record.get("pending_pages", 0) > 0 else "completed",
            )
            record = get_pdf_record(req.pdf_id)

        export_path = export_index_json(
            pdf_id=req.pdf_id,
            record=record,
            index_items=classified_final,
            indexed_start=indexed_start,
            indexed_end=indexed_end,
            total_pages=total_pages,
            index_source=index_source,
        )

    timing_collector.log_summary("index_generation")

    return {
        "index": classified_final,
        "total_pages": total_pages,
        "indexed_page_start": indexed_start,
        "indexed_page_end": indexed_end,
        "indexed_pages": total_chunks,
        "toc_items": len(index_items),
        "auto_items": len(auto_items),
        "index_source": index_source,
        "status": record["status"] if record else "index_ready",
        "retrieval_status": record["retrieval_status"] if record else "legacy",
        "pending_pages": record["pending_pages"] if record else max(total_pages - total_chunks, 0),
        "chat_ready": record["chat_ready"] if record else True,
        "index_export_file": str(export_path),
    }

# -- 4. GET PAGE TEXT ──────────────────────────────────────────────────────────
@app.get("/api/page-text/{pdf_id}/{page_num}")
async def get_page_text(pdf_id: str, page_num: int):
    """Get the stored text for a specific page."""
    cached = get_cached_pages(pdf_id, start_page=page_num, end_page=page_num)
    if cached:
        page = cached[0]
        return {
            "page_num": page_num,
            "text": page["text"],
            "metadata": {
                "page_num": page["page_num"],
                "used_ocr": page["used_ocr"],
                "vision_used": page["vision_used"],
                "handwriting_suspected": page["handwriting_suspected"],
                "extraction_method": page["extraction_method"],
                "stage": page["stage"],
            },
        }

    try:
        collection = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        raise HTTPException(404, "PDF not found")

    result = collection.get(
        ids=[f"{pdf_id}_p{page_num}"],
        include=["documents", "metadatas"],
    )
    if not result["documents"]:
        raise HTTPException(404, f"Page {page_num} not found")

    return {
        "page_num": page_num,
        "text": result["documents"][0],
        "metadata": result["metadatas"][0],
    }


@app.post("/api/ingest-batch")
async def ingest_batch_pdfs(
    files: list[UploadFile] = File(...),
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
    vectorize_now: bool = Form(False),
):
    results = []
    seen_pdf_ids: set[str] = set()
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "skipped", "reason": "Only PDF files are accepted"})
            continue
        pdf_bytes = await file.read()
        pdf_id = pdf_id_from_bytes(pdf_bytes)
        if pdf_id in seen_pdf_ids:
            results.append({
                "pdf_id": pdf_id,
                "filename": file.filename,
                "status": "skipped",
                "reason": "Duplicate PDF in the same batch. Skipping.",
                "skipped_duplicate": True,
            })
            continue
        seen_pdf_ids.add(pdf_id)
        existing = build_existing_pdf_payload(pdf_id, file.filename)
        if existing:
            results.append(existing)
            continue
        payload = await build_stage_one_payload_async(pdf_bytes, file.filename, start_page=start_page, end_page=end_page)
        payload["index_entries"] = len(payload.get("index", []))
        if vectorize_now and payload.get("pending_pages", 0) > 0:
            deferred_resp = await process_pending_pdf(payload["pdf_id"])
            payload.update({
                "status": deferred_resp.get("status", payload["status"]),
                "retrieval_status": deferred_resp.get("retrieval_status", payload["retrieval_status"]),
                "pending_pages": deferred_resp.get("pending_pages", payload["pending_pages"]),
                "chat_ready": deferred_resp.get("chat_ready", payload["chat_ready"]),
            })
        results.append(payload)
    return {"pdfs": results, "count": len(results)}


# -- 5. LIST INGESTED PDFs ─────────────────────────────────────────────────────
@app.get("/api/pdfs")
async def list_pdfs(search: str = ""):
    """List PDFs from the workflow state store."""
    return {"pdfs": list_pdf_records(search)}


@app.get("/api/pdf-status/{pdf_id}")
async def get_pdf_status(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    return record


@app.get("/api/pdfs/search")
async def search_pdfs(query: str = ""):
    return {"pdfs": list_pdf_records(query)}


@app.get("/api/queues")
async def get_queues():
    return {
        **build_queue_snapshot(),
        "runner": dict(deferred_runner_status),
        "index_runner": dict(index_runner_status),
    }


@app.post("/api/queues/reset")
async def reset_queue(queue_name: str = Form(...)):
    queue_value = (queue_name or "").strip().lower()
    if queue_value not in {"index", "deferred"}:
        raise HTTPException(400, "queue_name must be index or deferred")

    reset_count = 0
    for record in list_pdf_records():
        pending_pages = int(record.get("pending_pages") or 0)
        index_ready = bool(record.get("index_ready"))
        retrieval_status = record.get("retrieval_status") or ""
        status = record.get("status") or ""

        if queue_value == "index":
            should_reset = (not index_ready) and status in {"toc_scanned", "indexing_running", "failed", "needs_review"}
            if not should_reset:
                continue
            update_pdf_record(
                record["pdf_id"],
                status="toc_scanned",
                retrieval_status="pending_deferred_ingestion" if pending_pages > 0 else retrieval_status,
                queue_bucket="deferred" if pending_pages > 0 else "library",
                deferred_decision="pending" if pending_pages > 0 else record.get("deferred_decision", "completed"),
                last_error="",
            )
            reset_count += 1
            continue

        should_reset = pending_pages > 0 and (
            retrieval_status in {"full_ingestion_running", "queued_for_full_ingestion", "pending_deferred_ingestion", "failed"}
            or status == "full_ingestion_running"
        )
        if not should_reset:
            continue
        update_pdf_record(
            record["pdf_id"],
            status="index_ready" if index_ready else "toc_scanned",
            retrieval_status="pending_deferred_ingestion",
            chat_ready=False,
            queue_bucket="deferred",
            deferred_decision="queue" if record.get("deferred_decision") == "queue" else "pending",
            last_error="",
        )
        reset_count += 1

    return {
        "queue_name": queue_value,
        "reset_count": reset_count,
        "queues": build_queue_snapshot(),
    }


@app.get("/api/pdfs/{pdf_id}")
async def get_pdf_details(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    return {
        "pdf": record,
        "index": get_saved_index(pdf_id),
    }


@app.get("/api/pdfs/{pdf_id}/file")
async def download_pdf_file(pdf_id: str):
    pdf_path = stored_pdf_path(pdf_id)
    record = get_pdf_record(pdf_id)
    if not pdf_path.exists() or not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    return FileResponse(path=pdf_path, media_type="application/pdf", filename=record.get("filename") or f"{pdf_id}.pdf")


@app.post("/api/pdfs/{pdf_id}/deferred-choice")
async def set_deferred_choice(pdf_id: str, choice: str = Form(...)):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    choice_value = (choice or "").strip().lower()
    if choice_value not in {"queue", "skip"}:
        raise HTTPException(400, "Choice must be queue or skip")
    update_pdf_record(
        pdf_id,
        deferred_decision=choice_value,
        queue_bucket="deferred" if choice_value == "queue" else "library",
    )
    return {"pdf_id": pdf_id, "deferred_decision": choice_value}


def process_pending_pdf_impl(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    cached_pages = get_cached_pages(pdf_id)
    cached_page_numbers = {page["page_num"] for page in cached_pages}
    pending_page_numbers = [page for page in range(1, record["total_pages"] + 1) if page not in cached_page_numbers]
    if not pending_page_numbers:
        update_pdf_record(pdf_id, status="vectorized", retrieval_status="vectorized", chat_ready=True, pending_pages=0, queue_bucket="library", deferred_decision="completed")
        return {
            "pdf_id": pdf_id,
            "status": "vectorized",
            "retrieval_status": "vectorized",
            "pending_pages": 0,
            "processed_pages": 0,
            "chat_ready": True,
        }

    pdf_path = stored_pdf_path(pdf_id)
    if not pdf_path.exists():
        raise HTTPException(404, f"Stored PDF for {pdf_id} not found")

    update_pdf_record(pdf_id, status="full_ingestion_running", retrieval_status="full_ingestion_running")
    timing_collector = PdfTimingCollector(pdf_id, record.get("filename", ""))
    with timing_collector.stage("file open", "file_open"):
        doc = fitz.open(pdf_path)
    try:
        with timing_collector.stage("full text extraction", "full_text_extraction_time"):
            pages_data, stats = extract_pages_from_document(doc, pending_page_numbers, record["total_pages"], dpi=250, timing_collector=timing_collector)
    finally:
        doc.close()

    upsert_extracted_pages(pdf_id, pages_data, stage="deferred_ingestion")
    upsert_collection_pages(pdf_id, record["filename"], pages_data, reset=False, timing_collector=timing_collector)

    updated_indexed_pages = len(get_cached_pages(pdf_id))
    update_pdf_record(
        pdf_id,
        indexed_pages=updated_indexed_pages,
        status="vectorized",
        retrieval_status="vectorized",
        chat_ready=True,
        pending_pages=0,
        queue_bucket="library",
        deferred_decision="completed",
        last_error="",
    )

    timing_collector.log_summary("deferred_vectorization")

    return {
        "pdf_id": pdf_id,
        "status": "vectorized",
        "retrieval_status": "vectorized",
        "pending_pages": 0,
        "processed_pages": len(pages_data),
        "indexed_pages": updated_indexed_pages,
        "ocr_pages": stats["ocr_pages"],
        "vision_ocr_pages": stats["vision_ocr_pages"],
        "handwriting_suspected_pages": stats["handwriting_suspected_pages"],
        "chat_ready": True,
    }

def run_deferred_queue_worker(pdf_ids: list[str]):
    deferred_runner_status.update({
        "running": True,
        "processed": 0,
        "total": len(pdf_ids),
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": time.time(),
        "paused": False,
    })
    completed = 0
    try:
        for index, pdf_id in enumerate(pdf_ids, start=1):
            if deferred_runner_status.get("pause_requested"):
                deferred_runner_status.update({
                    "paused": True,
                    "processed": completed,
                    "current_pdf_id": "",
                    "current_filename": "",
                    "heartbeat_ts": time.time(),
                })
                break
            record = get_pdf_record(pdf_id) or {}
            deferred_runner_status.update({
                "current_pdf_id": pdf_id,
                "current_filename": record.get("filename", ""),
                "processed": completed,
                "heartbeat_ts": time.time(),
            })
            try:
                process_pending_pdf_impl(pdf_id)
                completed += 1
                deferred_runner_status["processed"] = completed
                deferred_runner_status["heartbeat_ts"] = time.time()
            except Exception as exc:
                log.exception("Deferred queue failed for %s", pdf_id)
                update_pdf_record(pdf_id, retrieval_status="failed", last_error=str(exc))
                deferred_runner_status["last_error"] = str(exc)
                deferred_runner_status["heartbeat_ts"] = time.time()
        else:
            deferred_runner_status["processed"] = len(pdf_ids)
            deferred_runner_status["heartbeat_ts"] = time.time()
    finally:
        deferred_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
        })


@app.post("/api/process-pending/{pdf_id}")
async def process_pending_pdf(pdf_id: str):
    return process_pending_pdf_impl(pdf_id)


@app.post("/api/process-pending")
async def process_pending_batch():
    results = []
    for pdf_id in list_pending_pdf_ids():
        results.append(process_pending_pdf_impl(pdf_id))
    return {"processed": results, "count": len(results)}


@app.post("/api/process-pending-runner")
async def process_pending_background():
    with deferred_runner_lock:
        if deferred_runner_status.get("running"):
            return {
                "started": False,
                "runner": dict(deferred_runner_status),
                "message": "Deferred queue is already running.",
            }
        pdf_ids = list_pending_pdf_ids()
        if not pdf_ids:
            return {
                "started": False,
                "runner": dict(deferred_runner_status),
                "message": "No PDFs are waiting in the deferred queue.",
            }
        deferred_runner_status.update({
            "pause_requested": False,
            "paused": False,
            "last_error": deferred_runner_status.get("last_error", ""),
        })
        worker = Thread(target=run_deferred_queue_worker, args=(pdf_ids,), daemon=True)
        worker.start()
    return {
        "started": True,
        "runner": dict(deferred_runner_status),
        "count": len(pdf_ids),
    }


@app.post("/api/process-pending-runner/control")
async def control_process_pending_runner(action: str = Form(...)):
    action_value = (action or "").strip().lower()
    if action_value not in {"stop", "resume"}:
        raise HTTPException(400, "action must be stop or resume")

    with deferred_runner_lock:
        if action_value == "stop":
            if deferred_runner_status.get("running"):
                deferred_runner_status.update({
                    "pause_requested": True,
                    "heartbeat_ts": time.time(),
                })
                return {
                    "accepted": True,
                    "message": "Stop requested. The queue will pause after the current PDF finishes.",
                    "runner": dict(deferred_runner_status),
                }
            if deferred_runner_status.get("paused"):
                return {
                    "accepted": True,
                    "message": "Deferred queue is already paused.",
                    "runner": dict(deferred_runner_status),
                }
            return {
                "accepted": False,
                "message": "Deferred queue is not running.",
                "runner": dict(deferred_runner_status),
            }

        if deferred_runner_status.get("running"):
            return {
                "accepted": False,
                "message": "Deferred queue is already running.",
                "runner": dict(deferred_runner_status),
            }
        pdf_ids = list_pending_pdf_ids()
        if not pdf_ids:
            deferred_runner_status.update({"paused": False, "pause_requested": False, "heartbeat_ts": time.time()})
            return {
                "accepted": False,
                "message": "No PDFs are waiting in the deferred queue.",
                "runner": dict(deferred_runner_status),
            }
        deferred_runner_status.update({
            "pause_requested": False,
            "paused": False,
            "processed": 0,
            "total": len(pdf_ids),
            "heartbeat_ts": time.time(),
        })
        worker = Thread(target=run_deferred_queue_worker, args=(pdf_ids,), daemon=True)
        worker.start()
        return {
            "accepted": True,
            "message": "Deferred queue resumed.",
            "runner": dict(deferred_runner_status),
        }


# -- 6. DELETE PDF ─────────────────────────────────────────────────────────────
@app.delete("/api/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    """Remove a PDF and all its vectors from the database."""
    try:
        chroma_client.delete_collection(f"pdf_{pdf_id}")
        delete_pdf_state(pdf_id)
        try:
            stored_pdf_path(pdf_id).unlink(missing_ok=True)
        except TypeError:
            if stored_pdf_path(pdf_id).exists():
                stored_pdf_path(pdf_id).unlink()
        return {"status": "deleted", "pdf_id": pdf_id}
    except Exception:
        raise HTTPException(404, f"PDF {pdf_id} not found")


# 7. TEXT TRANSFORM
@app.post("/api/text-transform")
async def text_transform(req: TextTransformRequest):
    source_text = (req.text or "").strip()
    if not source_text:
        raise HTTPException(400, "Text is required")

    action = (req.action or "").strip().lower()
    if action not in {"translate", "transliterate"}:
        raise HTTPException(400, "Action must be 'translate' or 'transliterate'")

    if action == "translate":
        system_prompt = """You are an expert bilingual legal document assistant.
Translate the provided court-document text into clear English.
Rules:
- Preserve names, dates, case numbers, exhibit labels, and legal references accurately.
- Do not summarize or omit content.
- Keep the meaning faithful to the original.
- If the input already contains English, keep it natural and preserve its meaning.
Return only the translated text."""
        user_prompt = f"Translate this text into English:\n\n{source_text}"
    else:
        system_prompt = """You are an expert in Hindi transliteration for legal documents.
Transliterate Devanagari/Hindi text into readable Roman script.
Rules:
- Preserve meaning only through transliteration, not translation.
- Keep English text, numbers, dates, and legal references as they are.
- Do not summarize or explain.
Return only the transliterated text."""
        user_prompt = f"Transliterate this text into Roman script without translating it:\n\n{source_text}"

    transformed = call_local_text(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1800,
        temperature=0.0,
    ).strip()

    return {
        "action": action,
        "text": transformed,
    }


# ── STARTUP ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_workflow_db()
    if not LOCAL_TEXT_MODEL:
        log.warning("LOCAL_TEXT_MODEL is not set - AI features will fail")
    log.info(f"Local LLM URL: {LOCAL_LLM_BASE_URL}")
    log.info(f"Vision model : {LOCAL_VISION_MODEL}")
    log.info(f"Text model   : {LOCAL_TEXT_MODEL}")
    log.info(f"ChromaDB path: {CHROMA_DB_PATH}")
    log.info("Server ready")
