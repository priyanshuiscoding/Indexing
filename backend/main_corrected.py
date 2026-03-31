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

import fitz                        # PyMuPDF
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
    init_db as init_workflow_db,
    list_pdf_records,
    list_pending_pdf_ids,
    replace_extracted_pages,
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


NVIDIA_API_KEY   = normalize_api_key(os.getenv("NVIDIA_API_KEY", ""))
VISION_MODEL     = os.getenv("VISION_MODEL", "mistralai/mistral-small-3.1-24b-instruct-2503")
TEXT_MODEL       = os.getenv("TEXT_MODEL",   "mistralai/mistral-small-3.1-24b-instruct-2503")
NVIDIA_BASE_URL  = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "./stored_pdfs")
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

# ── NVIDIA client ──────────────────────────────────────────────────────────────
nvidia_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY,
)

# ── Embedding model (local, offline, Hindi+English) ───────────────────────────
embedder = None
embedder_lock = Lock()

# ── ChromaDB ──────────────────────────────────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
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

def pdf_id_from_bytes(data: bytes) -> str:
    """Stable ID for a PDF based on its content hash."""
    return hashlib.md5(data).hexdigest()[:16]


def stored_pdf_path(pdf_id: str) -> Path:
    """Local on-disk copy of the uploaded PDF for later page-image reprocessing."""
    return Path(PDF_STORAGE_PATH) / f"{pdf_id}.pdf"


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
    if not ENABLE_HANDWRITTEN_HINDI_ASSIST or not NVIDIA_API_KEY:
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
    """
    Use page images for TOC extraction with improved structured table detection.
    Now focuses specifically on extracting table rows with Sr.No, Title, Page columns.
    """
    if not page_nums or not pdf_path.exists() or not NVIDIA_API_KEY:
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
            
            # IMPROVEMENT: Higher DPI for table text clarity
            image = render_page_image(doc[page_num - 1], dpi=300)
            image_b64 = image_to_jpeg_base64(image, max_side=2400, quality=90)
            
            # IMPROVED PROMPT: Specific to structured table extraction
            prompt = f"""You are reading an INDEX / TABLE OF CONTENTS page from an Indian court file.

This page contains a STRUCTURED TABLE with these columns:
- Sr. No. (Serial Number) - numbered 1, 2, 3... or sometimes क्रम.
- Particulars of the Documents (main title/description column)
- Annexure (optional reference column like "C-1", "A-2")
- Page No / Page Number (page number or range on the right)

EXTRACTION RULES:
1. Extract EVERY complete table row in order (Sr.No 1, 2, 3...)
2. Preserve original text exactly (Hindi Devanagari, English, mixed)
3. If a title spans multiple lines within one table row, combine them
4. If Sr.No blank, infer from sequence (after Sr.No 5, next is 6)
5. Page numbers: Convert Hindi digits (०-९) to Arabic (0-9) for pageFrom/pageTo ONLY
6. If only ONE page shown, use same for pageFrom and pageTo
7. If range shown ("1-4", "1 to 4", "1_4"), extract start and end
8. Include Annexure column if visible, else empty string

TABLE STRUCTURE YOU'LL SEE:
- Header row: "Sr. No | Particulars of the Documents | Annexure | Page No"
- Data rows: "1 | Index | - | 1" or "२ | Description | C-१ | २-३"

RETURN ONLY JSON (no markdown, no text):
[
  {{
    "serialNo": "1",
    "title": "Index",
    "annexure": "",
    "pageFrom": 1,
    "pageTo": 1
  }},
  {{
    "serialNo": "2",
    "title": "Chronology of events",
    "annexure": "",
    "pageFrom": 2,
    "pageTo": 3
  }}
]

CRITICAL RULES:
- Return [] if NOT an INDEX/TOC table
- Extract ALL rows visible in the table
- Use exact original text (don't translate Hindi)
- No markdown backticks, no extra text, only JSON array"""

            raw = call_nvidia_vision(image_b64, "image/jpeg", prompt, max_tokens=3500)
            parsed = safe_json(raw)
            
            if isinstance(parsed, list) and parsed:
                # Validate and clean the extracted items
                valid_items = []
                for row in parsed:
                    if isinstance(row, dict) and row.get("title", "").strip():
                        # Ensure proper page number coercion
                        row["pageFrom"] = coerce_page_number(row.get("pageFrom"), page_num)
                        row["pageTo"] = coerce_page_number(row.get("pageTo"), row.get("pageFrom", page_num))
                        row.setdefault("source", "toc-image")
                        row.setdefault("annexure", "")
                        row.setdefault("serialNo", "")
                        valid_items.append(row)
                
                if valid_items:
                    toc_items.extend(valid_items)
                    log.info(f"✓ TOC image extraction: page {page_num} extracted {len(valid_items)} rows")
                else:
                    log.info(f"Vision returned list but no valid rows on page {page_num}")
            else:
                log.debug(f"No structured TOC found on page {page_num} (vision returned: {type(parsed)})")
                
    finally:
        doc.close()

    return toc_items

def call_nvidia_text(messages: list[dict], max_tokens: int = 2000, temperature: float = 0.1) -> str:
    """Call NVIDIA text model (no image). Used for RAG reasoning."""
    resp = nvidia_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def call_nvidia_vision(image_b64: str, media_type: str, prompt: str, max_tokens: int = 2000) -> str:
    """Call NVIDIA vision model with a single image (Mistral limit = 1 image)."""
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
    """Parse JSON from model output, stripping markdown fences."""
    text = re.sub(r"```json|```", "", text).strip()
    # Find first [ or { and parse from there
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = text.find(start_char)
        if idx != -1:
            try:
                return json.loads(text[idx:])
            except Exception:
                pass
    return None


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
        title = choose_parent_document(raw_title, preview, candidates, scored)
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
        "handwritten_hindi_assist": ENABLE_HANDWRITTEN_HINDI_ASSIST and bool(NVIDIA_API_KEY),
        "workflow_db": str(WORKFLOW_DB_PATH),
    }


# ── 1. INGEST PDF ─────────────────────────────────────────────────────────────
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
    pdf_id = pdf_id_from_bytes(pdf_bytes)
    stored_pdf_path(pdf_id).write_bytes(pdf_bytes)
    log.info("Stage 1 ingest for PDF: %s  id=%s", file.filename, pdf_id)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    if total_pages < 1:
        doc.close()
        raise HTTPException(400, "PDF has no pages")

    start_page = max(1, min(start_page, total_pages))
    default_end_page = min(total_pages, start_page + 4)
    end_page = default_end_page if end_page is None else min(end_page, total_pages)
    if start_page > end_page:
        doc.close()
        raise HTTPException(400, "Start page must be less than or equal to end page")

    selected_page_numbers = list(range(start_page, end_page + 1))
    pages_data, stats = extract_pages_from_document(doc, selected_page_numbers, total_pages, dpi=250)
    doc.close()

    replace_extracted_pages(pdf_id, pages_data, stage="fast_index")
    upsert_collection_pages(pdf_id, file.filename, pages_data, reset=True)

    indexed_pages = len(pages_data)
    pending_pages = max(total_pages - indexed_pages, 0)
    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=file.filename,
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
    )

    log.info(
        "Stage 1 complete: indexed %s/%s pages, pending %s pages",
        indexed_pages,
        total_pages,
        pending_pages,
    )
    return {
        "pdf_id": pdf_id,
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
        "filename": file.filename,
    }


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


# -- 3. GENERATE INDEX ─────────────────────────────────────────────────────────
@app.post("/api/generate-index")
async def generate_index(req: IndexRequest):
    """
    Generate the structured document index from cached Stage 1 pages first.
    If a TOC is detected, ranges are expanded deterministically across the whole PDF.
    """
    record = get_pdf_record(req.pdf_id)
    all_pages = load_index_pages(req.pdf_id)
    if not all_pages:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_pages.sort(key=lambda item: item["page_num"])
    indexed_start = all_pages[0]["page_num"]
    indexed_end = all_pages[-1]["page_num"]
    total_pages = record["total_pages"] if record else indexed_end
    total_chunks = len(all_pages)

    # IMPROVED TOC CANDIDATE DETECTION
    toc_candidate_pages = []
    for page in all_pages:
        text = page["text"]
        lower_text = text.lower()

        # STRONG INDICATORS: Direct mentions of index/contents
        strong_match = bool(re.search(
            r'\bindex\b(?!\s+(?:page|number))|'   # "INDEX" but not "INDEX PAGE"
            r'\bcontents\b|'
            r'of\s+contents|'
            r'विषय\s*सूची|'                       # Hindi TOC
            r'अनुक्रमणिका|'                       # Hindi Index
            r'सूची',                              # Hindi List
            lower_text,
            re.IGNORECASE
        ))

        # STRUCTURAL INDICATORS: Table column headers
        structural_match = bool(re.search(
            r'sr\.?\s+no(?:\.)?|'
            r'क्रम\s*\.|'
            r'particulars?\s+of|'
            r'page\s+no(?:\.)?(?:\s|$)|'
            r'page\s+number|'
            r'annexure|'
            r'sheet\s+count',
            lower_text,
            re.IGNORECASE
        ))

        # TABLE STRUCTURE: Count rows that look like table entries
        lines = text.split('\n')
        table_row_count = 0
        for line in lines:
            # Patterns like: "1 Title 5" or Hindi digit equivalent
            if re.match(r'^\s*[०-९\d]+\s+.{10,200}[०-९\d]+\s*$', line.strip()):
                table_row_count += 1

        is_toc = strong_match or structural_match or (table_row_count >= 3)

        if is_toc:
            toc_candidate_pages.append(page)
            log.info(
                f"📋 TOC detected on page {page['page_num']}: "
                f"strong={strong_match}, structural={structural_match}, table_rows={table_row_count}"
            )

        if len(toc_candidate_pages) >= 8:
            break

    if toc_candidate_pages:
        log.info(
            f"Found {len(toc_candidate_pages)} candidate TOC pages: "
            f"{[p['page_num'] for p in toc_candidate_pages]}"
        )

    toc_page_nums = [p["page_num"] for p in toc_candidate_pages]

    index_items = []
    toc_source = ""
    toc_range_end = total_pages if record else indexed_end

    # TRY IMAGE EXTRACTION FIRST (most accurate for structured tables)
    if toc_page_nums:
        log.info(f"→ Attempting image-based extraction from pages {toc_page_nums}...")
        toc_image_items = extract_toc_from_page_images(stored_pdf_path(req.pdf_id), toc_page_nums)

        if isinstance(toc_image_items, list) and len(toc_image_items) >= 3:
            index_items = build_toc_ranges_from_items(
                toc_image_items, indexed_start, toc_range_end, "toc-image"
            )
            toc_source = "toc-image"
            log.info(
                f"✓ SUCCESS: Extracted {len(index_items)} index items "
                f"from structured TABLE OF CONTENTS"
            )

    # ONLY do OCR fallback if image extraction failed
    if not index_items and toc_candidate_pages:
        log.info("⚠ Image extraction yielded no results, trying OCR text fallback...")

        toc_pages_text = ""
        for page in toc_candidate_pages:
            toc_pages_text += f"\n--- Page {page['page_num']} ---\n{page['text']}\n"

        if toc_pages_text:
            toc_prompt = f"""You are reading OCR text from a TABLE OF CONTENTS / INDEX page.
These pages may contain a real table of contents for an Indian court file.

If a table of contents exists, extract the rows. Use both English and Hindi.
Preserve original title text (do not translate Hindi).
Read serial numbers and page numbers carefully.
Convert Hindi digits to Arabic numerals.
If one page number shown, set pageFrom and pageTo to same.

Extracted text from pages {toc_page_nums}:
{toc_pages_text[:8000]}

Return ONLY JSON (if not a TOC, return []):
[
  {{
    "serialNo": "1",
    "title": "exact original text",
    "pageFrom": 1,
    "pageTo": 4,
    "source": "toc-ocr"
  }}
]"""

            toc_raw = call_nvidia_text(
                messages=[{"role": "user", "content": toc_prompt}],
                max_tokens=3500,
            )
            toc_parsed = safe_json(toc_raw)
            if isinstance(toc_parsed, list) and len(toc_parsed) >= 3:
                index_items = build_toc_ranges_from_items(
                    toc_parsed, indexed_start, toc_range_end, "toc-ocr"
                )
                toc_source = "toc-ocr"
                log.info(f"✓ OCR fallback extracted {len(index_items)} items")

    scan_chunk = 25
    auto_items = []
    uncovered_pages = all_pages if not index_items else []

    if not index_items:
        log.info(f"⚠ No TOC found. Scanning {len(uncovered_pages)} pages for document boundaries...")

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

            raw = call_nvidia_text(
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
    else:
        log.info("✓ Using TOC-based index (skipping auto-scan)")

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

    log.info(f"📊 Final index: {len(classified_final)} items from source '{index_source}'")

    if record:
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


# -- 5. LIST INGESTED PDFs ─────────────────────────────────────────────────────
@app.get("/api/pdfs")
async def list_pdfs():
    """List PDFs from the workflow state store."""
    return {"pdfs": list_pdf_records()}


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

    cached_pages = get_cached_pages(pdf_id)
    cached_page_numbers = {page["page_num"] for page in cached_pages}
    pending_page_numbers = [page for page in range(1, record["total_pages"] + 1) if page not in cached_page_numbers]
    if not pending_page_numbers:
        update_pdf_record(pdf_id, status="vectorized", retrieval_status="vectorized", chat_ready=True, pending_pages=0)
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
    doc = fitz.open(pdf_path)
    try:
        pages_data, stats = extract_pages_from_document(doc, pending_page_numbers, record["total_pages"], dpi=250)
    finally:
        doc.close()

    upsert_extracted_pages(pdf_id, pages_data, stage="deferred_ingestion")
    upsert_collection_pages(pdf_id, record["filename"], pages_data, reset=False)

    updated_indexed_pages = len(get_cached_pages(pdf_id))
    update_pdf_record(
        pdf_id,
        indexed_pages=updated_indexed_pages,
        status="vectorized",
        retrieval_status="vectorized",
        chat_ready=True,
        pending_pages=0,
    )

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


@app.post("/api/process-pending")
async def process_pending_batch():
    results = []
    for pdf_id in list_pending_pdf_ids():
        results.append(await process_pending_pdf(pdf_id))
    return {"processed": results, "count": len(results)}


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


# ── 7. NVIDIA PROXY (for vision calls from frontend) ─────────────────────────
@app.post("/api/nvidia-proxy")
async def nvidia_proxy(req: NvidiaProxyRequest):
    """
    Proxy for NVIDIA API calls from the frontend.
    Adds the API key server-side so it never goes to the browser.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {NVIDIA_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       req.model,
                "messages":    req.messages,
                "max_tokens":  req.max_tokens,
                "temperature": req.temperature,
            },
        )
    if not resp.is_success:
        raise HTTPException(resp.status_code, resp.text)
    return resp.json()


# ── STARTUP ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    init_workflow_db()
    if not NVIDIA_API_KEY:
        log.warning("NVIDIA_API_KEY is not set — AI features will fail")
    log.info(f"Vision model : {VISION_MODEL}")
    log.info(f"Text model   : {TEXT_MODEL}")
    log.info(f"ChromaDB path: {CHROMA_DB_PATH}")
    log.info("Server ready")
