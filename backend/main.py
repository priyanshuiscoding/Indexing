"""
Court File Indexer — RAG Backend
FastAPI server handling:
  - PDF ingestion (OCR + vectorization)
  - Semantic search / chatbot queries
  - Automatic index generation
  - NVIDIA API proxy (avoids browser CORS)
"""

import os
import re
import json
import math
import base64
import hashlib
import logging
import tempfile
from pathlib import Path
from io import BytesIO
from threading import Lock
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import httpx

from workflow_state import (
    DB_PATH as WORKFLOW_DB_PATH,
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


def normalize_api_key(raw_value: str) -> str:
    """Accept keys with accidental quotes or a leading 'Bearer ' prefix."""
    value = (raw_value or "").strip().strip("\"' ")
    if value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


NVIDIA_API_KEY = normalize_api_key(os.getenv("NVIDIA_API_KEY", ""))
VISION_MODEL = os.getenv("VISION_MODEL", "mistralai/mistral-small-3.1-24b-instruct-2503")
TEXT_MODEL = os.getenv("TEXT_MODEL", "mistralai/mistral-small-3.1-24b-instruct-2503")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "./stored_pdfs")
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "hin+eng")
ENABLE_HANDWRITTEN_HINDI_ASSIST = os.getenv("ENABLE_HANDWRITTEN_HINDI_ASSIST", "true").lower() != "false"

try:
    PARENT_DOCUMENT_CATALOG = json.loads(DOCUMENT_CATALOG_PATH.read_text(encoding="utf-8"))
except Exception:
    PARENT_DOCUMENT_CATALOG = []

PARENT_DOCUMENT_NAMES = list(
    dict.fromkeys(
        item["name"].strip()
        for item in PARENT_DOCUMENT_CATALOG
        if item.get("name") and str(item["name"]).strip()
    )
)
PARENT_DOCUMENT_EMBEDDINGS = None
GENERIC_PARENT_NAMES = {"other", "others"}

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
USING_DEFAULT_NVIDIA_ENDPOINT = NVIDIA_BASE_URL.rstrip("/") == DEFAULT_NVIDIA_BASE_URL
MODEL_API_READY = bool(NVIDIA_API_KEY) or not USING_DEFAULT_NVIDIA_ENDPOINT
MODEL_API_KEY_FOR_CLIENT = NVIDIA_API_KEY or ("local-no-key" if MODEL_API_READY else "")

# ── NVIDIA client ─────────────────────────────────────────────────────────────
nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=MODEL_API_KEY_FOR_CLIENT,
)

# ── Embedding model ───────────────────────────────────────────────────────────
embedder = None
embedder_lock = Lock()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Court File Indexer API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def pdf_id_from_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:16]


def stored_pdf_path(pdf_id: str) -> Path:
    return Path(PDF_STORAGE_PATH) / f"{pdf_id}.pdf"


def get_or_create_collection(pdf_id: str):
    name = f"pdf_{pdf_id}"
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def get_embedder():
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
                except Exception as exc:
                    embedder = False
                    log.exception("Falling back to lightweight local embeddings: %s", exc)
    return embedder


def fallback_embed_texts(texts: list[str], dims: int = 384) -> list[list[float]]:
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
    model = get_embedder()
    if model is False:
        return fallback_embed_texts(texts)
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vecs.tolist()


def ocr_page_image(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config="--psm 6")
        return text.strip()
    except Exception as exc:
        log.warning("Tesseract OCR failed: %s", exc)
        return ""


def render_page_image(page: fitz.Page, dpi: int = 250) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_jpeg_base64(image: Image.Image, max_side: int = 1800, quality: int = 80) -> str:
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
    ocr_stats = analyze_extracted_text(ocr_text)
    direct_stats = analyze_extracted_text(direct_text)
    likely_scan = direct_stats["chars"] < 30
    weak_ocr = ocr_stats["chars"] < 140 or ocr_stats["words"] < 24 or ocr_stats["line_count"] < 4
    mixed_noise = ocr_stats["digit_ratio"] > 0.22 and ocr_stats["words"] < 40
    low_script_signal = ocr_stats["devanagari_ratio"] < 0.02 and ocr_stats["ascii_ratio"] < 0.12
    return likely_scan and (weak_ocr or mixed_noise or low_script_signal)


def extract_handwritten_page_text(image: Image.Image, page_num: int) -> Optional[str]:
    if not ENABLE_HANDWRITTEN_HINDI_ASSIST or not MODEL_API_READY:
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
        text = call_nvidia_vision(image_b64, "image/jpeg", prompt, max_tokens=2200).strip()
        return text or None
    except Exception as exc:
        log.warning("Vision transcription failed for page %s: %s", page_num, exc)
        return None


def extract_page_content(page: fitz.Page, page_num: int, dpi: int = 200) -> dict:
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

    return {
        "text": final_text,
        "used_ocr": True,
        "vision_used": vision_used,
        "handwriting_suspected": handwriting_suspected,
        "extraction_method": extraction_method,
    }


def extract_toc_from_page_images(pdf_path: Path, page_nums: list[int]) -> list[dict]:
    if not page_nums or not pdf_path.exists() or not MODEL_API_READY:
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

            image = render_page_image(doc[page_num - 1], dpi=300)
            image_b64 = image_to_jpeg_base64(image, max_side=2400, quality=90)

            prompt = f"""You are reading an INDEX / TABLE OF CONTENTS page from an Indian court file.

This page contains a STRUCTURED TABLE with these columns:
- Sr. No. (Serial Number) - numbered 1, 2, 3... or sometimes क्रम.
- Particulars of the Documents (main title/description column)
- Annexure (optional reference column like "C-1", "A-2")
- Page No / Page Number (page number or range on the right)

EXTRACTION RULES:
1. Extract EVERY complete table row in order.
2. Preserve original text exactly (Hindi Devanagari, English, mixed).
3. If a title spans multiple lines within one table row, combine them.
4. If Sr.No blank, infer from sequence.
5. Page numbers: Convert Hindi digits (०-९) to Arabic (0-9) for pageFrom/pageTo ONLY.
6. If only ONE page shown, use same for pageFrom and pageTo.
7. If range shown ("1-4", "1 to 4", "1_4"), extract start and end.
8. Include Annexure column if visible, else empty string.

RETURN ONLY JSON:
[
  {{
    "serialNo": "1",
    "title": "Index",
    "annexure": "",
    "pageFrom": 1,
    "pageTo": 1
  }}
]

CRITICAL RULES:
- Return [] if NOT an INDEX/TOC table.
- Extract ALL rows visible in the table.
- Use exact original text.
- No markdown backticks, no extra text, only JSON array."""

            raw = call_nvidia_vision(image_b64, "image/jpeg", prompt, max_tokens=3500)
            parsed = safe_json(raw)

            if isinstance(parsed, list) and parsed:
                valid_items = []
                for row in parsed:
                    if isinstance(row, dict) and row.get("title", "").strip():
                        row["pageFrom"] = coerce_page_number(row.get("pageFrom"), page_num)
                        row["pageTo"] = coerce_page_number(row.get("pageTo"), row.get("pageFrom", page_num))
                        row.setdefault("source", "toc-image")
                        row.setdefault("annexure", "")
                        row.setdefault("serialNo", "")
                        valid_items.append(row)

                if valid_items:
                    toc_items.extend(valid_items)
                    log.info("TOC image extraction: page %s extracted %s rows", page_num, len(valid_items))
    finally:
        doc.close()

    return toc_items


def call_nvidia_text(messages: list[dict], max_tokens: int = 2000, temperature: float = 0.1) -> str:
    resp = nvidia_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_nvidia_vision(image_b64: str, media_type: str, prompt: str, max_tokens: int = 2000) -> str:
    resp = nvidia_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


def safe_json(text: str):
    text = re.sub(r"```json|```", "", text).strip()
    for start_char in ["[", "{"]:
        idx = text.find(start_char)
        if idx != -1:
            try:
                return json.loads(text[idx:])
            except Exception:
                pass
    return None


def coerce_page_number(value, fallback: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        devanagari_map = str.maketrans("०१२३४५६७८९", "0123456789")
        normalized = value.translate(devanagari_map)
        match = re.search(r"\d+", normalized)
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


def extract_pages_from_document(doc: fitz.Document, page_numbers: list[int], total_pages: int, dpi: int = 250) -> tuple[list[dict], dict]:
    pages_data = []
    ocr_count = 0
    vision_ocr_count = 0
    handwriting_count = 0

    for page_num in page_numbers:
        page = doc[page_num - 1]
        page_data = extract_page_content(page, page_num, dpi=dpi)
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
            "Page %s/%s - %s%s - %s chars",
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


def upsert_collection_pages(pdf_id: str, filename: str, pages_data: list[dict], reset: bool = False):
    if reset:
        try:
            chroma_client.delete_collection(f"pdf_{pdf_id}")
        except Exception:
            pass

    collection = get_or_create_collection(pdf_id)
    if not pages_data:
        return collection

    batch_size = 50
    for i in range(0, len(pages_data), batch_size):
        batch = pages_data[i:i + batch_size]
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
        embeddings = embed_texts(documents)
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

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
            "page_num": int(meta["page_num"]),
            "text": doc_text,
            "used_ocr": bool(meta.get("used_ocr")),
            "vision_used": bool(meta.get("vision_used")),
            "handwriting_suspected": bool(meta.get("handwriting_suspected")),
            "extraction_method": meta.get("extraction_method", "unknown"),
            "stage": "vectorized",
        })

    pages.sort(key=lambda item: item["page_num"])
    return pages


def load_all_pages_for_pdf(pdf_id: str) -> list[dict]:
    pages = load_collection_pages(pdf_id)
    if pages:
        return pages

    cached_pages = get_cached_pages(pdf_id)
    cached_pages.sort(key=lambda item: item["page_num"])
    return cached_pages


def get_toc_search_pages(all_pages: list[dict], start_page: int = 1, end_page: int = 10) -> list[dict]:
    return [page for page in all_pages if start_page <= page["page_num"] <= end_page]


def detect_toc_candidate_pages(pages: list[dict], max_candidates: int = 5) -> list[dict]:
    toc_candidate_pages = []

    for page in pages:
        text = page.get("text", "") or ""
        lower_text = text.lower()

        strong_match = bool(re.search(
            r"\bindex\b(?!\s+(?:page|number))|"
            r"\btable\s+of\s+contents\b|"
            r"\bcontents\b|"
            r"विषय\s*सूची|"
            r"अनुक्रमणिका|"
            r"सूची",
            lower_text,
            re.IGNORECASE,
        ))

        structural_match = bool(re.search(
            r"sr\.?\s*no(?:\.)?|"
            r"क्रम|"
            r"particulars?\s+of|"
            r"page\s+no(?:\.)?(?:\s|$)|"
            r"page\s+number|"
            r"annexure|"
            r"sheet\s+count",
            lower_text,
            re.IGNORECASE,
        ))

        lines = text.splitlines()
        table_row_count = 0
        for line in lines:
            cleaned = line.strip()
            if re.match(r"^\s*[०-९\d]+\s+.{6,250}[०-९\d]+\s*$", cleaned):
                table_row_count += 1

        is_toc = strong_match or structural_match or (table_row_count >= 3)
        if is_toc:
            toc_candidate_pages.append(page)
            log.info(
                "TOC candidate page=%s strong=%s structural=%s table_rows=%s",
                page["page_num"],
                strong_match,
                structural_match,
                table_row_count,
            )

        if len(toc_candidate_pages) >= max_candidates:
            break

    return toc_candidate_pages


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

    raw = call_nvidia_text(
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
        if item.get("documentType", "").lower() not in GENERIC_PARENT_NAMES:
            continue
        prev_item = smoothed[idx - 1] if idx > 0 else None
        next_item = smoothed[idx + 1] if idx + 1 < len(smoothed) else None
        prev_title = (prev_item or {}).get("documentType", "")
        next_title = (next_item or {}).get("documentType", "")
        if prev_item and next_item and prev_title == next_title and prev_title.lower() not in GENERIC_PARENT_NAMES:
            item["documentType"] = prev_title
        elif prev_item and prev_title.lower() not in GENERIC_PARENT_NAMES and item.get("source") == "gap":
            item["documentType"] = prev_title
        elif next_item and next_title.lower() not in GENERIC_PARENT_NAMES and item.get("source") == "gap":
            item["documentType"] = next_title
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
            and item.get("documentType") == prev.get("documentType")
        ):
            prev["pageTo"] = item["pageTo"]
            prev["verifiedPageTo"] = item.get("verifiedPageTo", item["pageTo"])
            if not prev.get("serialNo"):
                prev["serialNo"] = item.get("serialNo", "")
            if not prev.get("courtFee"):
                prev["courtFee"] = item.get("courtFee", "")
            continue
        merged.append(dict(item))
    return merged


def verify_index_items_with_vectors(pdf_id: str, index_items: list[dict], all_pages: list[dict], search_k: int = 8) -> list[dict]:
    if not index_items:
        return []

    try:
        collection = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        return index_items

    all_result = collection.get(include=["documents", "metadatas", "embeddings"])
    rows = []
    for doc_text, meta, emb in zip(
        all_result.get("documents", []),
        all_result.get("metadatas", []),
        all_result.get("embeddings", []),
    ):
        rows.append({
            "page_num": int(meta["page_num"]),
            "text": doc_text or "",
            "embedding": emb,
        })

    verified = []
    for idx, item in enumerate(index_items):
        raw_title = (item.get("displayTitle") or item.get("originalTitle") or item.get("title") or "").strip()
        title_query = raw_title or item.get("title", "")
        if not title_query:
            verified.append(item)
            continue

        q_vec = embed_texts([title_query])[0]
        scored = []
        for row in rows:
            lexical = lexical_overlap_score(title_query, row["text"])
            semantic = sum(a * b for a, b in zip(q_vec, row["embedding"]))
            score = (semantic * 2.0) + (lexical * 1.5)
            scored.append((score, row["page_num"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_hits = [p for s, p in scored[:search_k] if s > 0.35]

        toc_from = int(item.get("pageFrom", 1))
        toc_to = int(item.get("pageTo", toc_from))
        next_from = None
        if idx + 1 < len(index_items):
            next_from = int(index_items[idx + 1].get("pageFrom", toc_to + 1))

        verified_from = toc_from
        verified_to = toc_to
        confidence = 0.55
        verification_status = "toc_only"

        if top_hits:
            nearest = min(top_hits, key=lambda p: abs(p - toc_from))
            if abs(nearest - toc_from) <= 2:
                verified_from = nearest
                verification_status = "verified"
                confidence = 0.90
            else:
                verification_status = "weak_match"
                confidence = 0.65

        if next_from is not None and verified_from < next_from:
            verified_to = max(verified_from, next_from - 1)
        else:
            verified_to = max(verified_from, toc_to)

        verified.append({
            **item,
            "pageFrom": toc_from,
            "pageTo": toc_to,
            "verifiedPageFrom": verified_from,
            "verifiedPageTo": verified_to,
            "verificationStatus": verification_status,
            "verificationConfidence": confidence,
            "matchedPages": top_hits[:5],
        })

    return verified


def classify_index_to_parent_documents(index_items: list[dict], all_pages: list[dict]) -> list[dict]:
    if not index_items or not PARENT_DOCUMENT_NAMES:
        return index_items

    classified = []
    for item in index_items:
        raw_title = item.get("displayTitle") or item.get("originalTitle") or item.get("title", "")
        preview = build_segment_preview(
            all_pages,
            item.get("verifiedPageFrom", item.get("pageFrom", 1)),
            item.get("verifiedPageTo", item.get("pageTo", 1)),
        )
        scored = score_parent_documents(raw_title, preview)
        candidates = shortlist_parent_documents(raw_title, preview)
        document_type = choose_parent_document(raw_title, preview, candidates, scored)

        classified.append({
            **item,
            "title": raw_title,
            "displayTitle": raw_title,
            "originalTitle": raw_title,
            "documentType": document_type,
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


class NvidiaProxyRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 2000
    temperature: float = 0.1

# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"vision": VISION_MODEL, "text": TEXT_MODEL},
        "embedding_ready": embedder is not None,
        "handwritten_hindi_assist": ENABLE_HANDWRITTEN_HINDI_ASSIST and MODEL_API_READY,
        "workflow_db": str(WORKFLOW_DB_PATH),
    }


@app.post("/api/ingest")
async def ingest_pdf(
    file: UploadFile = File(...),
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    """
    Full-document ingestion pipeline.
    OCR/extract all pages first, vectorize all pages, then index generation can safely
    search TOC/index inside pages 1-10 while still verifying against the full document.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_bytes = await file.read()
    pdf_id = pdf_id_from_bytes(pdf_bytes)
    stored_pdf_path(pdf_id).write_bytes(pdf_bytes)
    log.info("Full ingest for PDF: %s id=%s", file.filename, pdf_id)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    if total_pages < 1:
        doc.close()
        raise HTTPException(400, "PDF has no pages")

    page_numbers = list(range(1, total_pages + 1))
    try:
        pages_data, stats = extract_pages_from_document(doc, page_numbers, total_pages, dpi=250)
    finally:
        doc.close()

    replace_extracted_pages(pdf_id, pages_data, stage="full_ingestion")
    upsert_collection_pages(pdf_id, file.filename, pages_data, reset=True)

    indexed_pages = len(pages_data)
    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=file.filename,
        total_pages=total_pages,
        selected_start_page=1,
        selected_end_page=min(10, total_pages),
        indexed_pages=indexed_pages,
        status="vectorized",
        retrieval_status="vectorized",
        index_ready=False,
        chat_ready=True,
        pending_pages=0,
        index_source="",
    )

    return {
        "pdf_id": pdf_id,
        "total_pages": total_pages,
        "indexed_pages": indexed_pages,
        "indexed_page_start": 1,
        "indexed_page_end": total_pages,
        "ocr_pages": stats["ocr_pages"],
        "vision_ocr_pages": stats["vision_ocr_pages"],
        "handwriting_suspected_pages": stats["handwriting_suspected_pages"],
        "digital_pages": stats["digital_pages"],
        "status": "vectorized",
        "retrieval_status": "vectorized",
        "pending_pages": 0,
        "chat_ready": True,
        "filename": file.filename,
    }


@app.post("/api/query")
async def query_pdf(req: QueryRequest):
    record = get_pdf_record(req.pdf_id)
    if record and not record.get("chat_ready"):
        raise HTTPException(409, "Chat/search will be available after ingestion finishes for this PDF.")

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
    top_rows = [row for row in scored_rows[:max(3, min(req.top_k, len(scored_rows)))] if row["score"] > 0]
    if not top_rows:
        top_rows = scored_rows[:min(req.top_k, len(scored_rows))]

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

    answer = call_nvidia_text(
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


@app.post("/api/generate-index")
async def generate_index(req: IndexRequest):
    """
    Generate structured index after full-document vectorization.
    TOC/index discovery is restricted to pages 1-10.
    Extracted TOC rows are then verified against the full vectorized PDF.
    """
    record = get_pdf_record(req.pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {req.pdf_id} metadata not found. Please ingest this PDF again.")

    all_pages = load_all_pages_for_pdf(req.pdf_id)
    if not all_pages:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_pages.sort(key=lambda item: item["page_num"])
    total_pages = record["total_pages"]
    total_chunks = len(all_pages)

    toc_search_pages = get_toc_search_pages(all_pages, 1, min(10, total_pages))
    toc_candidate_pages = detect_toc_candidate_pages(toc_search_pages, max_candidates=5)
    toc_page_nums = [p["page_num"] for p in toc_candidate_pages]

    index_items = []
    toc_source = ""

    if toc_page_nums:
        log.info("Attempting TOC image extraction from pages %s", toc_page_nums)
        toc_image_items = extract_toc_from_page_images(stored_pdf_path(req.pdf_id), toc_page_nums)

        if isinstance(toc_image_items, list) and len(toc_image_items) >= 2:
            index_items = build_toc_ranges_from_items(
                toc_image_items,
                indexed_start=1,
                range_end=total_pages,
                default_source="toc-image",
            )
            toc_source = "toc-image"

    if not index_items and toc_candidate_pages:
        toc_pages_text = ""
        for page in toc_candidate_pages:
            toc_pages_text += f"\n--- Page {page['page_num']} ---\n{page['text']}\n"

        toc_prompt = f"""You are reading OCR text from an Indian court-file index/table-of-contents page.

Extract every table row in order.
Preserve original Hindi/English title exactly.
Convert Hindi digits to Arabic numerals only for page numbers.
Return ONLY JSON array. Return [] if this is not a real TOC/index.

Text:
{toc_pages_text[:12000]}

Format:
[
  {{
    "serialNo": "1",
    "title": "exact original text",
    "pageFrom": 1,
    "pageTo": 1,
    "source": "toc-ocr"
  }}
]"""

        toc_raw = call_nvidia_text(
            messages=[{"role": "user", "content": toc_prompt}],
            max_tokens=3500,
            temperature=0.0,
        )
        toc_parsed = safe_json(toc_raw)
        if isinstance(toc_parsed, list) and len(toc_parsed) >= 2:
            index_items = build_toc_ranges_from_items(
                toc_parsed,
                indexed_start=1,
                range_end=total_pages,
                default_source="toc-ocr",
            )
            toc_source = "toc-ocr"

    if not index_items:
        raise HTTPException(
            422,
            "No usable table of contents/index found in pages 1-10. Please review PDF or run manual indexing."
        )

    verified_items = verify_index_items_with_vectors(req.pdf_id, index_items, all_pages)
    classified_final = classify_index_to_parent_documents(verified_items, all_pages)
    index_source = toc_source or "toc"
    save_index(req.pdf_id, classified_final)
    update_pdf_record(
        req.pdf_id,
        status="index_ready",
        index_ready=True,
        index_source=index_source,
    )
    record = get_pdf_record(req.pdf_id)

    return {
        "index": classified_final,
        "total_pages": total_pages,
        "indexed_page_start": 1,
        "indexed_page_end": total_pages,
        "indexed_pages": total_chunks,
        "toc_search_window": [1, min(10, total_pages)],
        "toc_candidate_pages": toc_page_nums,
        "toc_items": len(index_items),
        "auto_items": 0,
        "index_source": index_source,
        "status": record["status"],
        "retrieval_status": record["retrieval_status"],
        "pending_pages": record["pending_pages"],
        "chat_ready": record["chat_ready"],
    }

@app.get("/api/index/{pdf_id}")
async def get_saved_index_route(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    saved = get_saved_index(pdf_id)
    if saved is None:
        raise HTTPException(404, f"No saved index found for PDF {pdf_id}")

    return {
        "pdf_id": pdf_id,
        "filename": record.get("filename", ""),
        "index": saved,
        "total_entries": len(saved),
        "index_ready": record.get("index_ready", False),
        "index_source": record.get("index_source", ""),
    }

@app.get("/api/page-text/{pdf_id}/{page_num}")
async def get_page_text(pdf_id: str, page_num: int):
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


@app.get("/api/pdfs")
async def list_pdfs():
    return {"pdfs": list_pdf_records()}


@app.get("/api/queues")
async def get_queues():
    return {
        "index_ready": [],
        "stage1_batch": [],
        "pending_vectorization": [],
        "vectorized": [],
        "reindex_review": [],
        "errors": [],
        "runner": {
            "running": False,
            "processed": 0,
            "total": 0,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": "",
            "pause_requested": False,
            "paused": False,
            "heartbeat_ts": 0,
        },
        "index_runner": {
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": "",
            "finished_pdf_id": "",
            "finished_filename": "",
            "status": "idle",
        },
        "stage1_batch_runner": {
            "running": False,
            "processed": 0,
            "total": 0,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": "",
            "heartbeat_ts": 0,
            "status": "idle",
        },
        "audit_runner": {
            "running": False,
            "processed": 0,
            "total": 0,
            "flagged": 0,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": "",
            "heartbeat_ts": 0,
            "status": "idle",
        },
        "reindex_runner": {
            "running": False,
            "processed": 0,
            "total": 0,
            "fixed": 0,
            "current_pdf_id": "",
            "current_filename": "",
            "last_error": "",
            "heartbeat_ts": 0,
            "status": "idle",
        },
    }


@app.get("/api/batch-reports")
async def get_batch_reports(limit: int = 8):
    return {"reports": [], "limit": max(1, min(limit, 100))}


@app.get("/api/pdf-status/{pdf_id}")
async def get_pdf_status(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    return record


@app.post("/api/process-pending/{pdf_id}")
async def process_pending_pdf(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    update_pdf_record(
        pdf_id,
        status="vectorized",
        retrieval_status="vectorized",
        chat_ready=True,
        pending_pages=0,
    )

    updated = get_pdf_record(pdf_id)
    return {
        "pdf_id": pdf_id,
        "status": updated["status"],
        "retrieval_status": updated["retrieval_status"],
        "pending_pages": updated["pending_pages"],
        "processed_pages": 0,
        "indexed_pages": updated["indexed_pages"],
        "chat_ready": updated["chat_ready"],
    }


@app.post("/api/process-pending")
async def process_pending_batch():
    results = []
    for pdf_id in list_pending_pdf_ids():
        results.append(await process_pending_pdf(pdf_id))
    return {"processed": results, "count": len(results)}


@app.delete("/api/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
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


@app.post("/api/nvidia-proxy")
async def nvidia_proxy(req: NvidiaProxyRequest):
    headers = {
        "Content-Type": "application/json",
    }
    if NVIDIA_API_KEY:
        headers["Authorization"] = f"Bearer {NVIDIA_API_KEY}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers=headers,
            json={
                "model": req.model,
                "messages": req.messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            },
        )

    if not resp.is_success:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


@app.on_event("startup")
async def startup():
    init_workflow_db()
    if not MODEL_API_READY:
        log.warning("Model API credentials are not configured for the default NVIDIA endpoint - AI features will fail")
    log.info("Vision model : %s", VISION_MODEL)
    log.info("Text model   : %s", TEXT_MODEL)
    log.info("ChromaDB path: %s", CHROMA_DB_PATH)
    log.info("Server ready")
