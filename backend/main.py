"""
Court File Indexer â€” RAG Backend  (v5.0 â€” Full Local Pipeline)
==============================================================

Environment (.env):
  LOCAL_LLM_BASE_URL              = http://127.0.0.1:11434   (Ollama base, /v1 auto-appended)
  LOCAL_TEXT_MODEL                = qwen2.5:14b
  LOCAL_VISION_MODEL              = qwen2.5vl:7b
  LOCAL_LLM_TIMEOUT               = 600
  ENABLE_HANDWRITTEN_HINDI_ASSIST = true
  CHROMA_DB_PATH                  = ./chroma_db
  PDF_STORAGE_PATH                = ./stored_pdfs
  INDEX_EXPORT_PATH               = ./index_exports
  TESSERACT_LANG                  = hin+eng
  DATABASE_URL                    = postgresql://postgres:post123@localhost:5432/court_rag
  WORKFLOW_SQLITE_PATH            = ./workflow.db

Pipeline (100 % local â€” no cloud, no NVIDIA API):
  Upload PDF
    â†’ OCR every page   (PyMuPDF direct â†’ Tesseract â†’ qwen2.5vl if handwritten)
    â†’ Vectorize pages  (sentence-transformers  â†’  ChromaDB)
    â†’ Persist text     (PostgreSQL via workflow_state.py)

  Generate Index   â† pure local, no LLM at all
    â†’ Detect TOC in pages 1-10          (tight regex heuristics)
    â†’ Parse TOC rows                     (regex, two-pass)
    â†’ Build / forward-fill page ranges
    â†’ Verify with full-doc vectors       (local embeddings)
    â†’ Classify document types            (alias map + local embeddings)
    â†’ Save to DB + export JSON

  Chat / Query
    â†’ Hybrid retrieval  (semantic + lexical + proximity)
    â†’ qwen2.5:14b answer generation  (local Ollama)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import httpx

from workflow_state import (
    STORAGE_BACKEND,
    STORAGE_TARGET,
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
    upsert_pdf_record,
    build_queue_snapshot,
)

# â”€â”€ Bootstrap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

# â”€â”€ HF cache â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HF_CACHE_ROOT = (
    Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir()) / "court-rag-hf-cache"
)
HF_CACHE_PATH = str(HF_CACHE_ROOT)
for _k in ("HF_HOME", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
    os.environ.setdefault(_k, HF_CACHE_PATH)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
Path(HF_CACHE_PATH).mkdir(parents=True, exist_ok=True)

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH",    "./chroma_db")
PDF_STORAGE_PATH  = os.getenv("PDF_STORAGE_PATH",  "./stored_pdfs")
INDEX_EXPORT_PATH = os.getenv("INDEX_EXPORT_PATH", "./index_exports")
TESSERACT_LANG    = os.getenv("TESSERACT_LANG",    "hin+eng")

# Ollama endpoint â€” /v1 appended automatically if missing
_RAW_BASE          = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
LOCAL_LLM_BASE_URL = _RAW_BASE if _RAW_BASE.endswith("/v1") else f"{_RAW_BASE}/v1"
LOCAL_TEXT_MODEL   = os.getenv("LOCAL_TEXT_MODEL",   "qwen2.5:14b")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
LOCAL_LLM_TIMEOUT  = int(os.getenv("LOCAL_LLM_TIMEOUT", "600"))
ENABLE_VISION      = os.getenv("ENABLE_HANDWRITTEN_HINDI_ASSIST", "true").lower() != "false"

# Document-type catalog
DOCUMENT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "document_catalog.json"
try:
    _CATALOG_RAW = json.loads(DOCUMENT_CATALOG_PATH.read_text(encoding="utf-8"))
except Exception:
    _CATALOG_RAW = []

PARENT_DOCUMENT_NAMES: list[str] = list(
    dict.fromkeys(
        item["name"].strip()
        for item in _CATALOG_RAW
        if item.get("name") and str(item["name"]).strip()
    )
)
GENERIC_PARENT_NAMES  = {"other", "others"}
_PARENT_EMBEDDINGS: Optional[list] = None

# â”€â”€ Storage dirs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
for _d in (CHROMA_DB_PATH, PDF_STORAGE_PATH, INDEX_EXPORT_PATH):
    Path(_d).mkdir(parents=True, exist_ok=True)

# â”€â”€ ChromaDB â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)

# â”€â”€ Ollama clients (OpenAI-compatible) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_http = httpx.Client(timeout=LOCAL_LLM_TIMEOUT)

_text_client   = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key="ollama", http_client=_http)
_vision_client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key="ollama", http_client=_http)

# â”€â”€ Embedding model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_embedder      = None
_embedder_lock = Lock()

# â”€â”€ FastAPI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
app = FastAPI(title="Court File Indexer API", version="5.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# EMBEDDING
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    log.info("Loading embedding model â€¦")
                    _embedder = SentenceTransformer(
                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        cache_folder=HF_CACHE_PATH,
                    )
                    log.info("Embedding model ready")
                except Exception as exc:
                    log.exception("Embedding load failed, using hash fallback: %s", exc)
                    _embedder = False
    return _embedder


def _fallback_embed(texts: list[str], dims: int = 384) -> list[list[float]]:
    vectors = []
    for text in texts:
        vec    = [0.0] * dims
        tokens = re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) or ["_"]
        for tok in tokens:
            idx     = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % dims
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    if model is False:
        return _fallback_embed(texts)
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOCAL LLM CALLS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def call_text_llm(
    messages: list[dict], max_tokens: int = 2000, temperature: float = 0.1
) -> str:
    """qwen2.5:14b â€” used ONLY for /api/query chat answers."""
    try:
        resp = _text_client.chat.completions.create(
            model=LOCAL_TEXT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Text LLM call failed: %s", exc)
        return f"[LLM unavailable: {exc}]"


def call_vision_llm(image_b64: str, prompt: str, max_tokens: int = 2200) -> str:
    """
    qwen2.5vl:7b â€” used ONLY during ingestion for handwritten / poor-OCR pages.
    Never called during index generation.
    """
    try:
        resp = _vision_client.chat.completions.create(
            model=LOCAL_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type":      "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Vision LLM call failed: %s", exc)
        return ""


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PDF / OCR / IMAGE HELPERS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def pdf_id_from_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()[:16]


def stored_pdf_path(pdf_id: str) -> Path:
    return Path(PDF_STORAGE_PATH) / f"{pdf_id}.pdf"


def render_page_image(page: fitz.Page, dpi: int = 250) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def image_to_jpeg_b64(img: Image.Image, max_side: int = 1800, quality: int = 80) -> str:
    img = img.copy()
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS  # Pillow < 9
    img.thumbnail((max_side, max_side), resample)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ocr_page_image(image: Image.Image, psm: int = 6) -> str:
    try:
        return pytesseract.image_to_string(
            image, lang=TESSERACT_LANG, config=f"--psm {int(psm)}"
        ).strip()
    except Exception as exc:
        log.warning("Tesseract OCR failed: %s", exc)
        return ""


def _text_stats(text: str) -> dict:
    chars = len(text)
    words = len(re.findall(r"\w+", text, flags=re.UNICODE))
    lines = len([l for l in text.splitlines() if l.strip()])
    dev   = len(re.findall(r"[\u0900-\u097F]", text))
    asc   = len(re.findall(r"[A-Za-z]", text))
    digs  = len(re.findall(r"\d", text))
    return {
        "chars": chars, "words": words, "lines": lines,
        "dev_ratio": dev  / max(chars, 1),
        "asc_ratio": asc  / max(chars, 1),
        "dig_ratio": digs / max(chars, 1),
    }


def _needs_vision(direct: str, ocr: str) -> bool:
    if not ENABLE_VISION:
        return False
    ds  = _text_stats(direct)
    os_ = _text_stats(ocr)
    return (
        ds["chars"] < 30
        and (
            os_["chars"] < 140
            or os_["words"] < 24
            or os_["lines"] < 4
            or (os_["dig_ratio"] > 0.22 and os_["words"] < 40)
            or (os_["dev_ratio"] < 0.02 and os_["asc_ratio"] < 0.12)
        )
    )


_VISION_PROMPT = (
    "You are transcribing a scanned Indian court-file page.\n"
    "The page may contain handwritten or printed Hindi (Devanagari) and/or English.\n\n"
    "Rules:\n"
    "- Preserve Hindi in Devanagari exactly.\n"
    "- Preserve English exactly.\n"
    "- Keep helpful line breaks.\n"
    "- Do NOT summarize, translate, or explain.\n"
    "- Include headings, labels, serials, names, dates, table rows.\n"
    "- If a word is unclear, give your best reading; do not skip it.\n\n"
    "Return ONLY the transcription text, nothing else."
)


def extract_page_content(page: fitz.Page, page_num: int, dpi: int = 250) -> dict:
    """
    Text extraction priority:
      1. PyMuPDF direct text  (digital PDFs)
      2. Tesseract OCR        (scanned pages)
      3. Vision LLM assist    (handwritten / very poor OCR, if ENABLE_VISION=true)
    """
    direct = page.get_text("text").strip()
    if len(direct) > 40:
        return {
            "text":                  re.sub(r"\n{3,}", "\n\n", direct),
            "used_ocr":              False,
            "vision_used":           False,
            "handwriting_suspected": False,
            "extraction_method":     "digital",
        }

    image    = render_page_image(page, dpi=dpi)
    ocr_text = ocr_page_image(image)

    vision_used           = False
    handwriting_suspected = _needs_vision(direct, ocr_text)

    if handwriting_suspected:
        vision_text = call_vision_llm(image_to_jpeg_b64(image), _VISION_PROMPT)
        if vision_text:
            if _text_stats(vision_text)["chars"] >= max(_text_stats(ocr_text)["chars"], 80):
                ocr_text    = vision_text
                vision_used = True

    return {
        "text":                  ocr_text or f"[Page {page_num} â€” no readable text]",
        "used_ocr":              True,
        "vision_used":           vision_used,
        "handwriting_suspected": handwriting_suspected,
        "extraction_method":     "vision_ocr" if vision_used else "ocr",
    }


def extract_pages_from_document(
    doc: fitz.Document,
    page_numbers: list[int],
    total_pages: int,
    dpi: int = 250,
) -> tuple[list[dict], dict]:
    pages_data:               list[dict] = []
    ocr_n = vision_n = hw_n = 0

    for pn in page_numbers:
        pd   = extract_page_content(doc[pn - 1], pn, dpi=dpi)
        text = pd["text"] or f"[Page {pn} â€” no readable text]"
        if pd["used_ocr"]:   ocr_n    += 1
        if pd["vision_used"]:vision_n += 1
        if pd["handwriting_suspected"]: hw_n += 1

        pages_data.append({
            "page_num": pn,       "text":    text,
            "used_ocr": pd["used_ocr"],       "vision_used":           pd["vision_used"],
            "handwriting_suspected": pd["handwriting_suspected"],
            "extraction_method":     pd["extraction_method"],
        })
        log.info(
            "Page %s/%s â€” %-12s â€” %s chars",
            pn, total_pages, pd["extraction_method"], len(text),
        )

    return pages_data, {
        "ocr_pages":                   ocr_n,
        "vision_ocr_pages":            vision_n,
        "handwriting_suspected_pages": hw_n,
        "digital_pages":               len(page_numbers) - ocr_n,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# CHROMA
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def get_or_create_collection(pdf_id: str):
    name = f"pdf_{pdf_id}"
    try:
        return chroma_client.get_collection(name)
    except Exception:
        return chroma_client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_collection_pages(
    pdf_id: str, filename: str, pages_data: list[dict], reset: bool = False
):
    if reset:
        try:
            chroma_client.delete_collection(f"pdf_{pdf_id}")
        except Exception:
            pass
    col = get_or_create_collection(pdf_id)
    for i in range(0, len(pages_data), 50):
        batch = pages_data[i : i + 50]
        col.upsert(
            ids       = [f"{pdf_id}_p{p['page_num']}" for p in batch],
            documents = [p["text"]                    for p in batch],
            metadatas = [{
                "page_num":              p["page_num"],
                "used_ocr":              p["used_ocr"],
                "vision_used":           p["vision_used"],
                "handwriting_suspected": p["handwriting_suspected"],
                "extraction_method":     p["extraction_method"],
                "filename":              filename,
            } for p in batch],
            embeddings = embed_texts([p["text"] for p in batch]),
        )
    return col


def load_collection_pages(pdf_id: str) -> list[dict]:
    try:
        col = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        return []
    result = col.get(include=["documents", "metadatas"])
    pages = [
        {
            "page_num":              int(m["page_num"]),
            "text":                  d,
            "used_ocr":              bool(m.get("used_ocr")),
            "vision_used":           bool(m.get("vision_used")),
            "handwriting_suspected": bool(m.get("handwriting_suspected")),
            "extraction_method":     m.get("extraction_method", "unknown"),
            "stage":                 "vectorized",
        }
        for d, m in zip(result.get("documents", []), result.get("metadatas", []))
    ]
    pages.sort(key=lambda x: x["page_num"])
    return pages


def load_all_pages_for_pdf(pdf_id: str) -> list[dict]:
    pages = load_collection_pages(pdf_id)
    if pages:
        return pages
    cached = get_cached_pages(pdf_id)
    cached.sort(key=lambda x: x["page_num"])
    return cached


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TOC DETECTION  â€”  pure local regex, NO LLM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_DEVA_MAP = str.maketrans("०१२३४५६७८९", "0123456789")

def _to_arabic(s: str) -> str:
    return s.translate(_DEVA_MAP)


# Header-level signals: explicit "index" / "table of contents" or Hindi equivalents
_STRONG_HEADER_RE = re.compile(
    r"(?:"
    r"\bindex\b(?!\s*(?:page|no|number|finger))|"
    r"\btable\s+of\s+contents\b|"
    r"विषय\s*सूची|अनुक्रमणिका|(?<!\w)सूची(?!\w)"
    r")",
    re.IGNORECASE,
)

# Structural column-header signals
_STRUCTURAL_RE = re.compile(
    r"(?:"
    r"sr\.?\s*no\.?|क्रम\s*(?:सं(?:ख्या)?)?|"
    r"particulars?\s+of\s+(?:the\s+)?documents?|"
    r"(?:page|pg)\.?\s*no\.?(?:\s|$)|page\s+number|"
    r"annexure|sheet\s+count|दस्तावेज|अनुलग्न"
    r")",
    re.IGNORECASE,
)

# Table-body row: serial  text  page-number at end
_TABLE_ROW_RE = re.compile(
    r"^\s*[०-९\d]+[\.\)\/]?\s+.{5,200}[०-९\d]+\s*$"
)


_TOC_HEADER_HINTS = (
    "index",
    "table of contents",
    "toc",
    "chronology",
    "chronology of events",
    "list of documents",
    "index & chronology",
    "\u0935\u093f\u0937\u092f",
    "\u0938\u0942\u091a\u0940",
    "\u0905\u0928\u0941\u0915\u094d\u0930\u092e\u0923\u093f\u0915\u093e",
)

_TOC_STRUCT_HINTS = (
    "sr",
    "serial",
    "particular",
    "annexure",
    "page",
    "pg",
    "sheet",
    "document",
    "\u0915\u094d\u0930\u092e",
    "\u0926\u0938\u094d\u0924\u093e\u0935\u0947\u091c",
    "\u0905\u0928\u0941\u0932\u0917\u094d\u0928",
)

_TOC_SEED_TEXTS = [
    "Table of contents with serial number particulars annexure and page number",
    "Index page listing documents and page ranges for a court file",
    "Index and chronology of events",
    "\u0928\u094d\u092f\u093e\u092f\u093e\u0932\u092f \u092a\u094d\u0930\u0915\u0930\u0923 \u0915\u0940 \u0935\u093f\u0937\u092f \u0938\u0942\u091a\u0940 \u091c\u093f\u0938\u092e\u0947\u0902 \u0915\u094d\u0930\u092e \u0938\u0902\u0916\u094d\u092f\u093e \u0926\u0938\u094d\u0924\u093e\u0935\u0947\u091c \u0914\u0930 \u092a\u0943\u0937\u094d\u0920 \u0938\u0902\u0916\u094d\u092f\u093e \u0926\u093f\u090f \u0939\u094b\u0902",
]
_TOC_SEED_EMBEDDINGS: Optional[list[list[float]]] = None

_SERIAL_START_RE = re.compile(r"^\s*(?P<serial>[०-९\d]{1,4})[\.\)\/:-]?\s+(?P<body>.+)$")
_PAGE_TAIL_RE = re.compile(
    r"(?P<page>\d{1,4}(?:\s*(?:-|to)\s*\d{1,4})?/?)(?:\s*)$",
    re.IGNORECASE,
)
_OCR_CHAR_FIX = str.maketrans({
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
})


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _get_toc_seed_embeddings() -> list[list[float]]:
    global _TOC_SEED_EMBEDDINGS
    if _TOC_SEED_EMBEDDINGS is None:
        _TOC_SEED_EMBEDDINGS = embed_texts(_TOC_SEED_TEXTS)
    return _TOC_SEED_EMBEDDINGS


def _normalize_toc_line(line: str) -> str:
    line = _to_arabic((line or "").translate(_OCR_CHAR_FIX))
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"\b(?:t0|t0o)\b", "to", line, flags=re.IGNORECASE)
    return line


def _extract_toc_row_from_line(line: str, fallback_page: int = 1) -> Optional[dict]:
    line = _normalize_toc_line(line)
    if not line or _SKIP_LINE_RE.match(line):
        return None

    serial = ""
    body = line
    sm = _SERIAL_START_RE.match(line)
    if sm:
        serial = _to_arabic(sm.group("serial").strip())
        body = sm.group("body").strip()

    pm = _PAGE_TAIL_RE.search(body)
    if not pm:
        return None

    page_s = pm.group("page").strip().rstrip("/")
    title_part = body[: pm.start()].strip(" .:-")
    if len(title_part) < 2:
        return None

    annexure = ""
    ann_m = _ANNEXURE_RE.search(title_part)
    if ann_m:
        annexure = ann_m.group(0).strip()
        title_part = (title_part[: ann_m.start()] + title_part[ann_m.end() :]).strip()

    if len(title_part) < 2:
        return None

    pf, pt = _parse_page_range(page_s, fallback_page)
    return {
        "serialNo": serial,
        "title": title_part,
        "annexure": annexure,
        "pageFrom": pf,
        "pageTo": pt,
        "source": "toc-stitch",
    }


def _parse_stitched_toc_rows(text: str, fallback_page: int = 1) -> list[dict]:
    lines = [_normalize_toc_line(l) for l in (text or "").splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return []

    stitched: list[str] = []
    cur = ""
    for line in lines:
        if _SERIAL_START_RE.match(line):
            if cur:
                stitched.append(cur.strip())
            cur = line
            continue
        if cur:
            cur = f"{cur} {line}".strip()
        else:
            stitched.append(line)
    if cur:
        stitched.append(cur.strip())

    rows: list[dict] = []
    for line in stitched:
        row = _extract_toc_row_from_line(line, fallback_page=fallback_page)
        if row:
            rows.append(row)
    return rows


def _parse_json_list(payload: str) -> list[dict]:
    if not payload:
        return []
    payload = payload.strip()
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", payload, flags=re.DOTALL | re.IGNORECASE)
    if m:
        payload = m.group(1)
    else:
        m = re.search(r"(\[.*\])", payload, flags=re.DOTALL)
        if m:
            payload = m.group(1)
    try:
        parsed = json.loads(payload)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        pf, pt = _parse_page_range(str(item.get("pageFrom", item.get("page", ""))), 1)
        if "pageTo" in item:
            _, pt2 = _parse_page_range(str(item.get("pageTo", "")), pt)
            pt = max(pf, pt2)
        out.append({
            "serialNo": str(item.get("serialNo", "")).strip(),
            "title": title,
            "annexure": str(item.get("annexure", "")).strip(),
            "pageFrom": pf,
            "pageTo": pt,
            "source": "toc-llm-json",
        })
    return out


def _toc_rows_from_local_llm(text: str, fallback_page: int = 1) -> list[dict]:
    prompt = (
        "Extract only table-of-contents/index rows from this OCR text. "
        "Return strict JSON array only, no prose. "
        "Schema for each row: "
        '{"serialNo":"", "title":"", "annexure":"", "pageFrom":1, "pageTo":1}. '
        "If not found, return [].\n\n"
        f"Text:\n{text[:9000]}"
    )
    raw = call_text_llm(
        [{"role": "user", "content": prompt}],
        max_tokens=1400,
        temperature=0.0,
    )
    rows = _parse_json_list(raw)
    if not rows:
        return []
    for row in rows:
        row["pageFrom"] = max(1, _coerce_int(row.get("pageFrom"), fallback_page))
        row["pageTo"] = max(row["pageFrom"], _coerce_int(row.get("pageTo"), row["pageFrom"]))
    return rows


def _toc_rows_from_vision_llm(image: Image.Image, fallback_page: int = 1) -> list[dict]:
    if not ENABLE_VISION:
        return []
    prompt = (
        "This is a possible court-file index/table-of-contents page. "
        "Extract rows and return strict JSON array only, no prose. "
        "Schema: "
        '[{"serialNo":"", "title":"", "annexure":"", "pageFrom":1, "pageTo":1}]. '
        "Accept page ranges and normalize numerals."
    )
    raw = call_vision_llm(image_to_jpeg_b64(image), prompt, max_tokens=1400)
    rows = _parse_json_list(raw)
    if not rows:
        return []
    for row in rows:
        row["pageFrom"] = max(1, _coerce_int(row.get("pageFrom"), fallback_page))
        row["pageTo"] = max(row["pageFrom"], _coerce_int(row.get("pageTo"), row["pageFrom"]))
    return rows


def _extract_toc_ocr_text_from_pdf(pdf_id: str, page_num: int, psm: int = 6) -> str:
    pdf_path = stored_pdf_path(pdf_id)
    if not pdf_path.exists():
        return ""
    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > doc.page_count:
                return ""
            img = render_page_image(doc[page_num - 1], dpi=300)
        finally:
            doc.close()
        return ocr_page_image(img, psm=psm)
    except Exception as exc:
        log.warning("TOC OCR fallback failed for p=%s: %s", page_num, exc)
        return ""


def detect_toc_candidate_pages(pages: list[dict], max_candidates: int = 5) -> list[dict]:
    """
    Hybrid TOC candidate ranking:
      - header keywords
      - structural keywords
      - row-shape count
      - embedding similarity against TOC seed prompts
    """
    if not pages:
        return []

    texts = [(_to_arabic((p.get("text", "") or "")[:12000])) for p in pages]
    page_embeddings: list[list[float]] = []
    seed_embeddings: list[list[float]] = []
    try:
        page_embeddings = embed_texts(texts)
        seed_embeddings = _get_toc_seed_embeddings()
    except Exception as exc:
        log.warning("TOC embedding scoring unavailable: %s", exc)

    ranked: list[tuple[float, dict, dict]] = []
    for idx, page in enumerate(pages):
        text = texts[idx]
        low = text.lower()
        strong = bool(_STRONG_HEADER_RE.search(text))
        struct = bool(_STRUCTURAL_RE.search(text))
        header_hits = sum(1 for kw in _TOC_HEADER_HINTS if kw in low)
        struct_hits = sum(1 for kw in _TOC_STRUCT_HINTS if kw in low)
        row_hits = sum(1 for ln in text.splitlines() if _TABLE_ROW_RE.match(ln.strip()))

        emb_score = 0.0
        if idx < len(page_embeddings) and seed_embeddings:
            emb_score = max(
                (_cosine_similarity(page_embeddings[idx], seed) for seed in seed_embeddings),
                default=0.0,
            )

        score = (
            (3.5 if strong else 0.0)
            + (2.0 if struct else 0.0)
            + min(header_hits, 4) * 1.5
            + min(struct_hits, 6) * 0.7
            + min(row_hits, 8) * 1.0
            + max(0.0, emb_score) * 4.0
        )
        dbg = {
            "strong": strong,
            "struct": struct,
            "header_hits": header_hits,
            "struct_hits": struct_hits,
            "row_hits": row_hits,
            "emb": round(emb_score, 4),
            "score": round(score, 3),
        }
        ranked.append((score, page, dbg))

    ranked.sort(key=lambda x: (x[0], x[1].get("page_num", 0)), reverse=True)
    candidates = [page for _score, page, _dbg in ranked[: max(1, max_candidates)]]

    for _score, page, dbg in ranked[: min(len(ranked), max(8, max_candidates))]:
        log.info("TOC score p=%s -> %s", page.get("page_num"), dbg)

    return candidates


def expand_toc_candidate_pages(
    candidates: list[dict],
    all_pages: list[dict],
    next_offsets: tuple[int, ...] = (1, 2),
) -> list[dict]:
    page_map = {int(p["page_num"]): p for p in all_pages}
    selected: set[int] = set()
    for page in candidates:
        pn = int(page["page_num"])
        selected.add(pn)
        selected.add(max(1, pn - 1))
        for off in next_offsets:
            selected.add(pn + int(off))
    return [page_map[pn] for pn in sorted(selected) if pn in page_map]


def _load_pdf_page_image(pdf_id: str, page_num: int, dpi: int = 300) -> Optional[Image.Image]:
    pdf_path = stored_pdf_path(pdf_id)
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > doc.page_count:
                return None
            return render_page_image(doc[page_num - 1], dpi=dpi)
        finally:
            doc.close()
    except Exception as exc:
        log.warning("Failed loading page image for TOC fallback p=%s: %s", page_num, exc)
        return None


def extract_toc_rows_with_fallback(pdf_id: str, page: dict) -> tuple[list[dict], str]:
    page_num = int(page["page_num"])
    base_text = page.get("text", "") or ""
    best_rows = parse_toc_rows_hybrid(base_text, fallback_page=page_num)
    best_method = "cached-hybrid"
    best_text = base_text

    if len(best_rows) < 2:
        ocr6 = _extract_toc_ocr_text_from_pdf(pdf_id, page_num, psm=6)
        if ocr6:
            rows6 = parse_toc_rows_hybrid(ocr6, fallback_page=page_num)
            if len(rows6) > len(best_rows):
                best_rows, best_method, best_text = rows6, "toc-ocr-psm6", ocr6

    if len(best_rows) < 2:
        ocr11 = _extract_toc_ocr_text_from_pdf(pdf_id, page_num, psm=11)
        if ocr11:
            rows11 = parse_toc_rows_hybrid(ocr11, fallback_page=page_num)
            if len(rows11) > len(best_rows):
                best_rows, best_method, best_text = rows11, "toc-ocr-psm11", ocr11

    if len(best_rows) < 2:
        llm_rows = _toc_rows_from_local_llm(best_text, fallback_page=page_num)
        if len(llm_rows) > len(best_rows):
            best_rows, best_method = llm_rows, "toc-text-llm"

    if len(best_rows) < 2 and ENABLE_VISION:
        img = _load_pdf_page_image(pdf_id, page_num, dpi=300)
        if img is not None:
            vision_rows = _toc_rows_from_vision_llm(img, fallback_page=page_num)
            if len(vision_rows) > len(best_rows):
                best_rows, best_method = vision_rows, "toc-vision-llm"

    return best_rows, best_method


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# TOC ROW PARSING  â€”  two-pass regex, NO LLM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_ANNEXURE_RE = re.compile(
    r"\b([A-Za-z]-?\d+|Annexure\s*[-\w]*|अनुलग्न\s*[-\d]*)\b", re.IGNORECASE
)

# Pass-1: strict â€”  <serial>  <title>  [<annexure>]  <page/range>
_ROW_STRICT = re.compile(
    r"^(?P<serial>[०-९\d]+[\.\)\/]?)\s+"
    r"(?P<rest>.{3,}?)\s{1,8}"
    r"(?P<page>[०-९\d]+(?:\s*(?:-|–|to)\s*[०-९\d]+)?)\s*$",
    re.UNICODE,
)

# Pass-2: loose â€” any text  <2+ spaces>  <page/range>
_ROW_LOOSE = re.compile(
    r"^(?P<title>.{4,}?)\s{2,}(?P<page>[०-९\d]+(?:\s*(?:-|–)\s*[०-९\d]+)?)\s*$"
)

_SKIP_LINE_RE = re.compile(
    r"^(?:sr\.?\s*no|page\s*no|particulars|क्रम|annexure|सं|index|"
    r"table\s+of\s+contents|विषय|सूची)\s*[:\.]?\s*$",
    re.IGNORECASE,
)


def _parse_page_range(raw: str, fallback: int) -> tuple[int, int]:
    raw = _to_arabic(raw.strip().rstrip("/"))
    raw = raw.replace("–", "-").replace("—", "-").replace("−", "-")
    raw = re.sub(r"\bto\b", "-", raw, flags=re.IGNORECASE)
    m   = re.search(r"(\d+)\s*-\s*(\d+)", raw)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)", raw)
    if m:
        n = int(m.group(1)); return n, n
    return fallback, fallback


def parse_toc_rows_from_text(text: str, fallback_page: int = 1) -> list[dict]:
    """
    Extract index rows from OCR text.
    Pass 1 (strict) â†’ if â‰¥2 rows found, return immediately.
    Pass 2 (loose)  â†’ fallback for less-structured layouts.
    """
    rows:  list[dict] = []
    lines: list[str]  = [l.rstrip() for l in text.splitlines()]

    # â”€â”€ Pass 1: strict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for line in lines:
        m = _ROW_STRICT.match(line.strip())
        if not m:
            continue
        serial = _to_arabic(m.group("serial").rstrip(".)/ "))
        rest   = m.group("rest").strip()
        page_s = m.group("page")

        ann_m    = _ANNEXURE_RE.search(rest)
        annexure = ann_m.group(0).strip() if ann_m else ""
        if ann_m:
            rest = (rest[: ann_m.start()] + rest[ann_m.end() :]).strip()

        title = rest.strip()
        if not title or len(title) < 2:
            continue

        pf, pt = _parse_page_range(page_s, fallback_page)
        rows.append({
            "serialNo": serial, "title": title, "annexure": annexure,
            "pageFrom": pf, "pageTo": pt, "source": "toc-regex-strict",
        })

    if len(rows) >= 2:
        return rows

    # â”€â”€ Pass 2: loose â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    rows = []
    for line in lines:
        line = line.strip()
        if not line or _SKIP_LINE_RE.match(line):
            continue
        m = _ROW_LOOSE.match(line)
        if not m:
            continue
        title = m.group("title").strip()
        if len(title) < 3:
            continue
        pf, pt = _parse_page_range(m.group("page"), fallback_page)
        rows.append({
            "serialNo": "", "title": title, "annexure": "",
            "pageFrom": pf, "pageTo": pt, "source": "toc-regex-loose",
        })

    return rows


def parse_toc_rows_hybrid(text: str, fallback_page: int = 1) -> list[dict]:
    """
    Higher-recall TOC parser:
      1) stitched multiline rows
      2) existing regex parser
      3) deduplicate by (title, pageFrom)
    """
    stitched = _parse_stitched_toc_rows(text, fallback_page=fallback_page)
    regex_rows = parse_toc_rows_from_text(text, fallback_page=fallback_page)
    merged = stitched + regex_rows

    out: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for item in merged:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        pf = _coerce_int(item.get("pageFrom"), fallback_page)
        key = (title.lower(), pf)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _coerce_int(value, fallback: int) -> int:
    if isinstance(value, int):   return value
    if isinstance(value, float): return int(value)
    if isinstance(value, str):
        m = re.search(r"\d+", _to_arabic(value))
        if m: return int(m.group())
    return fallback


def build_toc_ranges_from_items(
    items: list[dict], indexed_start: int, range_end: int, default_source: str
) -> list[dict]:
    """
    Normalize â†’ sort â†’ deduplicate â†’ forward-fill page ranges.
    Each item's pageTo = next item's pageFrom âˆ’ 1  (when unambiguous).
    """
    out:  list[dict]  = []
    seen: set[tuple]  = set()

    for item in items:
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        pf  = _coerce_int(item.get("pageFrom"), indexed_start)
        pt  = _coerce_int(item.get("pageTo"),   pf)
        pf  = max(indexed_start, min(pf, range_end))
        pt  = max(pf, min(pt, range_end))
        key = (title.lower(), pf)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "title":         title,
            "displayTitle":  str(item.get("displayTitle")  or title).strip(),
            "originalTitle": str(item.get("originalTitle") or title).strip(),
            "pageFrom":      pf,
            "pageTo":        pt,
            "source":        item.get("source", default_source),
            "serialNo":      str(item.get("serialNo",  "")),
            "annexure":      str(item.get("annexure",  "")),
            "courtFee":      str(item.get("courtFee",  "")),
        })

    out.sort(key=lambda x: (x["pageFrom"], x["title"]))

    # Forward-fill
    for i, item in enumerate(out):
        if i + 1 < len(out):
            nxt = out[i + 1]["pageFrom"]
            if nxt > item["pageFrom"]:
                item["pageTo"] = max(item["pageFrom"], nxt - 1)
        item["pageTo"] = min(item["pageTo"], range_end)

    return out


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# VECTOR VERIFICATION  â€”  local embeddings, NO LLM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if len(t) > 1]


def lexical_overlap(query: str, page_text: str) -> float:
    q_tok = tokenize(query)
    if not q_tok:
        return 0.0
    p_set   = set(tokenize(page_text))
    hits    = sum(1 for t in q_tok if t in p_set)
    density = hits / max(len(set(q_tok)), 1)
    phrase  = 2.5 if query.strip().lower() in (page_text or "").lower() else 0.0
    return hits + density + phrase


def verify_index_items_with_vectors(
    pdf_id: str,
    index_items: list[dict],
    all_pages: list[dict],
    search_k: int = 8,
) -> list[dict]:
    """
    Per TOC item: find the best matching page in the full vectorized document
    and adjust verifiedPageFrom / verifiedPageTo.  Fully local.
    """
    if not index_items:
        return []
    try:
        col = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        log.warning("Collection not found for %s â€” skipping verification", pdf_id)
        return index_items

    all_res = col.get(include=["documents", "metadatas", "embeddings"])
    rows = [
        {"page_num": int(m["page_num"]), "text": d, "emb": e}
        for d, m, e in zip(
            all_res.get("documents",  []),
            all_res.get("metadatas",  []),
            all_res.get("embeddings", []),
        )
    ]

    verified: list[dict] = []
    for idx, item in enumerate(index_items):
        title = (
            item.get("displayTitle") or item.get("originalTitle") or item.get("title") or ""
        ).strip()

        if not title:
            verified.append({
                **item,
                "verifiedPageFrom":       item.get("pageFrom", 1),
                "verifiedPageTo":         item.get("pageTo",   1),
                "verificationStatus":     "no_title",
                "verificationConfidence": 0.0,
                "matchedPages":           [],
            })
            continue

        q_vec  = embed_texts([title])[0]
        scored = sorted(
            [
                (
                    sum(a * b for a, b in zip(q_vec, r["emb"])) * 2.0
                    + lexical_overlap(title, r["text"]) * 1.5,
                    r["page_num"],
                )
                for r in rows
            ],
            reverse=True,
        )
        top_hits = [p for s, p in scored[:search_k] if s > 0.35]

        toc_from  = int(item.get("pageFrom", 1))
        toc_to    = int(item.get("pageTo",   toc_from))
        next_from = (
            int(index_items[idx + 1].get("pageFrom", toc_to + 1))
            if idx + 1 < len(index_items) else None
        )

        verified_from = toc_from
        status        = "toc_only"
        confidence    = 0.55

        if top_hits:
            nearest = min(top_hits, key=lambda p: abs(p - toc_from))
            if abs(nearest - toc_from) <= 2:
                verified_from = nearest
                status        = "verified"
                confidence    = 0.90
            else:
                status     = "weak_match"
                confidence = 0.65

        if next_from is not None and verified_from < next_from:
            verified_to = max(verified_from, next_from - 1)
        else:
            verified_to = max(verified_from, toc_to)

        verified.append({
            **item,
            "pageFrom":               toc_from,
            "pageTo":                 toc_to,
            "verifiedPageFrom":       verified_from,
            "verifiedPageTo":         verified_to,
            "verificationStatus":     status,
            "verificationConfidence": confidence,
            "matchedPages":           top_hits[:5],
        })

    return verified


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PARENT-DOCUMENT CLASSIFICATION  â€”  alias map + local embeddings, NO LLM
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_ALIAS_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(table\s+of\s+contents|index)\b|सूची|विषय.?सूची|अनुक्रमणिका", re.I), "Index"),
    (re.compile(r"vakalat|वकालतनामा",              re.I), "Vakalat Nama"),
    (re.compile(r"written\s+statement|लिखित",      re.I), "Written Statement"),
    (re.compile(r"\brejoinder\b",                   re.I), "Rejoinder"),
    (re.compile(r"\breply\b|जवाब",                 re.I), "Reply"),
    (re.compile(r"\breplication\b",                 re.I), "Replication"),
    (re.compile(r"affidavit|शपथ\s*पत्र",           re.I), "Affidavit"),
    (re.compile(r"power\s+of\s+attorney",           re.I), "Power of Attorney"),
    (re.compile(r"memo\s+of\s+parties",             re.I), "Memo of Parties"),
    (re.compile(r"list\s+of\s+dates|dates.+events", re.I), "List of Dates & Events"),
    (re.compile(r"brief\s+synopsis|synopsis",       re.I), "Brief Synopsis"),
    (re.compile(r"annexure|अनुलग्न|संलग्न",        re.I), "Annexure"),
    (re.compile(r"impugned\s+order",                re.I), "Impugned Order"),
    (re.compile(r"application|प्रार्थना\s*पत्र|अर्जी", re.I), "Application"),
    (re.compile(r"court\s+fee|stamp\s+paper|e-court", re.I), "e-Court Fee/Stamp Paper"),
    (re.compile(r"final\s+order|अंतिम\s+आदेश",    re.I), "FINAL ORDER"),
    (re.compile(r"office\s+note",                   re.I), "Office Note"),
    (re.compile(r"administrative\s+order",           re.I), "Administrative Orders"),
    (re.compile(r"\bnotice\b|सूचना",               re.I), "Notices"),
    (re.compile(r"\bletter\b",                      re.I), "Letter"),
    (re.compile(r"paper\s+book",                    re.I), "Paper Book"),
    (re.compile(r"\breport\b|प्रतिवेदन",            re.I), "Reports"),
    (re.compile(r"identity\s+proof|पहचान",          re.I), "Identity Proof"),
    (re.compile(r"process\s+fee",                   re.I), "Process Fee"),
    (re.compile(r"urgent\s+form|urgency",            re.I), "Urgent Form"),
    (re.compile(r"\bplaint\b|वाद\s*पत्र",           re.I), "Plaint"),
    (re.compile(r"\bpetition\b|याचिका",             re.I), "Petition"),
    (re.compile(r"order\s+sheet|आदेश\s*पत्र",      re.I), "Order Sheet"),
    (re.compile(r"\bchallan\b|चालान",               re.I), "Challan"),
    (re.compile(r"\bexhibit\b|प्रदर्श",             re.I), "Exhibit"),
    (re.compile(r"certificate|प्रमाण\s*पत्र",      re.I), "Certificate"),
    (re.compile(r"\bdecree\b|डिक्री",              re.I), "Decree"),
    (re.compile(r"judgment|judgement|निर्णय",        re.I), "Judgment"),
    (re.compile(r"\bsummons\b|समन",                re.I), "Summons"),
    (re.compile(r"\bwarrant\b|वारंट",              re.I), "Warrant"),
]


def _normalize_label(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/,()\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _direct_alias(title: str, preview: str) -> Optional[str]:
    combined = f"{title}\n{preview}"
    for pattern, target in _ALIAS_MAP:
        if pattern.search(combined) and target in PARENT_DOCUMENT_NAMES:
            return target
    norm_combined = _normalize_label(combined)
    for name in PARENT_DOCUMENT_NAMES:
        norm = _normalize_label(name)
        if norm and norm in norm_combined:
            return name
    return None


def _get_parent_embeddings():
    global _PARENT_EMBEDDINGS
    if _PARENT_EMBEDDINGS is None and PARENT_DOCUMENT_NAMES:
        _PARENT_EMBEDDINGS = embed_texts(PARENT_DOCUMENT_NAMES)
    return _PARENT_EMBEDDINGS or []


def _score_parent_docs(title: str, preview: str) -> list[tuple[float, str]]:
    if not PARENT_DOCUMENT_NAMES:
        return []
    seg   = f"{title}\n{preview}".strip()
    svec  = embed_texts([seg or title or "document"])[0]
    pvecs = _get_parent_embeddings()
    scored = []
    for name, nvec in zip(PARENT_DOCUMENT_NAMES, pvecs):
        lex   = lexical_overlap(name, seg)
        exact = 4.0 if _normalize_label(name) in _normalize_label(title) else 0.0
        prev  = 1.5 if _normalize_label(name) in _normalize_label(preview) else 0.0
        sem   = sum(a * b for a, b in zip(svec, nvec))
        gen_p = -3.0 if name.lower() in GENERIC_PARENT_NAMES else 0.0
        bonus = 2.0 if (name.lower() in title.lower()
                        and name.lower() not in GENERIC_PARENT_NAMES) else 0.0
        scored.append((sem * 2.8 + lex * 1.8 + exact + prev + gen_p + bonus, name))
    scored.sort(reverse=True)
    return scored


def _build_segment_preview(
    all_pages: list[dict], pf: int, pt: int, max_chars: int = 1200
) -> str:
    parts, total = [], 0
    for page in all_pages:
        if not (pf <= page["page_num"] <= pt):
            continue
        snippet = (page["text"] or "").strip()[:500]
        if not snippet:
            continue
        parts.append(f"[p{page['page_num']}] {snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(parts)


def classify_index_items(
    index_items: list[dict], all_pages: list[dict]
) -> list[dict]:
    """
    Assign documentType via:
      1. Static regex alias map  (~0 ms)
      2. Exact catalog-name match
      3. Local embedding similarity
    No LLM calls whatsoever.
    """
    if not index_items:
        return index_items

    result: list[dict] = []
    for item in index_items:
        title   = (
            item.get("displayTitle") or item.get("originalTitle") or item.get("title") or ""
        ).strip()
        preview = _build_segment_preview(
            all_pages,
            item.get("verifiedPageFrom", item.get("pageFrom", 1)),
            item.get("verifiedPageTo",   item.get("pageTo",   1)),
        )

        direct = _direct_alias(title, preview)
        if direct:
            doc_type = direct
        elif PARENT_DOCUMENT_NAMES:
            scored   = _score_parent_docs(title, preview)
            non_gen  = [(s, n) for s, n in scored if n.lower() not in GENERIC_PARENT_NAMES]
            pool     = non_gen or scored
            doc_type = pool[0][1] if pool else "Other"
        else:
            doc_type = "Other"

        result.append({
            **item,
            "title":         title,
            "displayTitle":  title,
            "originalTitle": title,
            "documentType":  doc_type,
        })

    return _merge_adjacent(result)


def _merge_adjacent(items: list[dict]) -> list[dict]:
    if not items:
        return items
    merged = [dict(items[0])]
    for item in items[1:]:
        prev = merged[-1]
        if (
            item.get("title")            == prev.get("title")
            and item.get("pageFrom")     == prev.get("pageTo", 0) + 1
            and item.get("documentType") == prev.get("documentType")
        ):
            prev["pageTo"]         = item["pageTo"]
            prev["verifiedPageTo"] = item.get("verifiedPageTo", item["pageTo"])
        else:
            merged.append(dict(item))
    return merged


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# JSON EXPORT  (index_exports/)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def export_index_json(pdf_id: str, filename: str, index: list[dict]) -> str:
    safe = re.sub(r"[^\w\-.]", "_", Path(filename).stem)[:60]
    out  = Path(INDEX_EXPORT_PATH) / f"{safe}_{pdf_id}.json"
    out.write_text(
        json.dumps(
            {"pdf_id": pdf_id, "filename": filename, "total_items": len(index), "index": index},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Index JSON exported â†’ %s", out)
    return str(out)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# REQUEST / RESPONSE MODELS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class QueryRequest(BaseModel):
    pdf_id:       str
    question:     str
    top_k:        int           = 8
    current_page: Optional[int] = None


class IndexRequest(BaseModel):
    pdf_id: str


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# ROUTES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.get("/health")
def health():
    return {
        "status":           "ok",
        "pipeline":         "local-only (Ollama)",
        "embedding_ready":  _embedder not in (None, False),
        "vision_assist":    ENABLE_VISION,
        "text_model":       LOCAL_TEXT_MODEL,
        "vision_model":     LOCAL_VISION_MODEL if ENABLE_VISION else "disabled",
        "llm_endpoint":     LOCAL_LLM_BASE_URL,
        "llm_timeout_s":    LOCAL_LLM_TIMEOUT,
        "parent_doc_types": len(PARENT_DOCUMENT_NAMES),
        "workflow_backend": STORAGE_BACKEND,
        "workflow_target":  STORAGE_TARGET,
    }


# â”€â”€ /api/ingest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/api/ingest")
async def ingest_pdf(
    file:       UploadFile    = File(...),
    start_page: int           = Form(1),
    end_page:   Optional[int] = Form(None),
):
    """
    Full ingest:
      1. OCR / extract all pages
      2. Vectorize â†’ ChromaDB
      3. Persist text â†’ PostgreSQL
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    pdf_bytes = await file.read()
    pdf_id    = pdf_id_from_bytes(pdf_bytes)
    stored_pdf_path(pdf_id).write_bytes(pdf_bytes)
    log.info("Ingest â€” file=%s  id=%s", file.filename, pdf_id)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    if total_pages < 1:
        doc.close()
        raise HTTPException(400, "PDF has no pages")

    selected_start = max(1, start_page)
    selected_end = min(end_page if end_page is not None else total_pages, total_pages)
    if selected_start > selected_end:
        doc.close()
        raise HTTPException(400, "Invalid page range")
    try:
        pages_data, stats = extract_pages_from_document(
            doc, list(range(1, total_pages + 1)), total_pages, dpi=250
        )
    finally:
        doc.close()

    replace_extracted_pages(pdf_id, pages_data, stage="full_ingestion")
    upsert_collection_pages(pdf_id, file.filename, pages_data, reset=True)

    upsert_pdf_record(
        pdf_id              = pdf_id,
        filename            = file.filename,
        total_pages         = total_pages,
        selected_start_page = selected_start,
        selected_end_page   = selected_end,
        indexed_pages       = len(pages_data),
        status              = "vectorized",
        retrieval_status    = "vectorized",
        index_ready         = False,
        chat_ready          = True,
        pending_pages       = 0,
        index_source        = "",
    )

    return {
        "pdf_id": pdf_id, "total_pages": total_pages,
        "indexed_pages": len(pages_data),
        "indexed_page_start": 1, "indexed_page_end": total_pages,
        **stats,
        "status": "vectorized", "retrieval_status": "vectorized",
        "pending_pages": 0, "chat_ready": True, "filename": file.filename,
    }


# â”€â”€ /api/generate-index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/api/generate-index")
async def generate_index(req: IndexRequest):
    """
    Pure-local index generation (no cloud):
      1. Detect TOC pages in 1-25   (hybrid ranking)
      2. Parse TOC rows              (stitching + OCR/LLM fallback)
      3. Build / forward-fill ranges
      4. Verify via local vectors
      5. Classify document types     (alias map + local embeddings)
      6. Save to DB + export JSON
    """
    record = get_pdf_record(req.pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_pages = load_all_pages_for_pdf(req.pdf_id)
    if not all_pages:
        raise HTTPException(404, f"No page data for PDF {req.pdf_id}. Please ingest first.")

    all_pages.sort(key=lambda x: x["page_num"])
    total_pages = record["total_pages"]
    toc_window_end = min(25, total_pages)
    toc_window = [p for p in all_pages if 1 <= p["page_num"] <= toc_window_end]

    # Step 1 - hybrid TOC ranking in pages 1..25
    top_candidates = detect_toc_candidate_pages(toc_window, max_candidates=6)
    candidate_nums = [int(p["page_num"]) for p in top_candidates]
    expanded_candidates = expand_toc_candidate_pages(top_candidates, all_pages)
    expanded_candidate_nums = [int(p["page_num"]) for p in expanded_candidates]
    log.info("TOC top candidates: %s", candidate_nums)
    log.info("TOC expanded candidates: %s", expanded_candidate_nums)

    # Step 2 - parse TOC rows with stitching + TOC-only OCR/LLM fallback
    raw_items: list[dict] = []
    for page in expanded_candidates:
        parsed, method = extract_toc_rows_with_fallback(req.pdf_id, page)
        if parsed:
            log.info("TOC rows p=%s -> %s via %s", page["page_num"], len(parsed), method)
            raw_items.extend(parsed)

    # Deduplicate across overlapping candidates
    seen_keys: set[tuple] = set()
    deduped:   list[dict] = []
    for item in raw_items:
        key = (item["title"].lower().strip(), item["pageFrom"])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(item)
    raw_items = deduped

    # Step 3 â€” build page ranges
    index_items: list[dict] = []
    if len(raw_items) >= 2:
        index_items = build_toc_ranges_from_items(
            raw_items, indexed_start=1, range_end=total_pages, default_source="toc-hybrid"
        )

    if not index_items:
        raise HTTPException(
            422,
            detail=(
                "No usable table of contents / index found in pages 1-25. "
                "Confirm the PDF has an index page, or add entries manually."
            ),
        )

    # Step 4 â€” vector verification
    verified = verify_index_items_with_vectors(req.pdf_id, index_items, all_pages)

    # Step 5 â€” classify document types
    classified = classify_index_items(verified, all_pages)

    # Step 6 â€” persist
    save_index(req.pdf_id, classified)
    update_pdf_record(
        req.pdf_id, status="index_ready", index_ready=True, index_source="toc-hybrid"
    )

    try:
        export_path = export_index_json(
            req.pdf_id, record.get("filename", ""), classified
        )
    except Exception as exc:
        log.warning("Index JSON export failed (non-fatal): %s", exc)
        export_path = ""

    record = get_pdf_record(req.pdf_id)
    return {
        "index":               classified,
        "total_pages":         total_pages,
        "indexed_page_start":  1,
        "indexed_page_end":    total_pages,
        "indexed_pages":       len(all_pages),
        "toc_search_window":   [1, toc_window_end],
        "toc_candidate_pages": candidate_nums,
        "toc_expanded_pages":  expanded_candidate_nums,
        "toc_items_parsed":    len(raw_items),
        "index_source":        "toc-hybrid",
        "export_path":         export_path,
        "status":              record["status"],
        "retrieval_status":    record["retrieval_status"],
        "pending_pages":       record["pending_pages"],
        "chat_ready":          record["chat_ready"],
    }


# â”€â”€ /api/query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.post("/api/query")
async def query_pdf(req: QueryRequest):
    """Hybrid retrieval + qwen2.5:14b answer generation."""
    record = get_pdf_record(req.pdf_id)
    if record and not record.get("chat_ready"):
        raise HTTPException(409, "Chat will be available after ingestion finishes.")

    try:
        col = chroma_client.get_collection(f"pdf_{req.pdf_id}")
    except Exception:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_res = col.get(include=["documents", "metadatas", "embeddings"])
    rows = [
        {"page_num": int(m["page_num"]), "text": d, "emb": e}
        for d, m, e in zip(
            all_res["documents"], all_res["metadatas"], all_res["embeddings"]
        )
    ]

    q_vec  = embed_texts([req.question])[0]
    q_toks = tokenize(req.question)

    scored = []
    for row in rows:
        sem  = sum(a * b for a, b in zip(q_vec, row["emb"]))
        lex  = lexical_overlap(req.question, row["text"])
        prox = 0.0
        if req.current_page is not None:
            diff = abs(row["page_num"] - req.current_page)
            prox = 3.0 if diff == 0 else (1.5 if diff <= 2 else 0.0)
        tp   = (
            sum(1 for t in q_toks if t in (row["text"] or "").lower())
            / max(len(q_toks), 1)
        ) if q_toks else 0.0
        scored.append({**row, "score": sem * 2.0 + lex + prox + tp})

    scored.sort(key=lambda r: (r["score"], r["page_num"]), reverse=True)
    top_k = max(3, min(req.top_k, len(scored)))
    top   = [r for r in scored[:top_k] if r["score"] > 0] or scored[:top_k]

    context   = "\n\n".join(
        f"--- Page {r['page_num']} ---\n{r['text'][:1400]}" for r in top
    )
    page_refs = sorted({r["page_num"] for r in top})

    answer = call_text_llm([
        {
            "role": "system",
            "content": (
                "You are an expert assistant for Indian court documents. "
                "Documents may contain Hindi (Devanagari) and English. "
                "Answer ONLY from the provided pages. "
                "Always cite the page number(s) your answer comes from. "
                "Keep Hindi in Devanagari. "
                "If the answer is not in the pages, say so clearly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {req.question}\n\n"
                f"Relevant pages:\n{context}\n\n"
                "Answer with page citations."
            ),
        },
    ])

    return {"answer": answer, "page_refs": page_refs, "chunks_used": len(top)}


# â”€â”€ Standard CRUD / status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/index/{pdf_id}")
async def get_saved_index_route(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    saved = get_saved_index(pdf_id)
    if saved is None:
        raise HTTPException(404, f"No saved index for PDF {pdf_id}")
    return {
        "pdf_id": pdf_id, "filename": record.get("filename", ""),
        "index": saved, "total_entries": len(saved),
        "index_ready": record.get("index_ready", False),
        "index_source": record.get("index_source", ""),
    }


@app.get("/api/page-text/{pdf_id}/{page_num}")
async def get_page_text(pdf_id: str, page_num: int):
    cached = get_cached_pages(pdf_id, start_page=page_num, end_page=page_num)
    if cached:
        pg = cached[0]
        return {"page_num": page_num, "text": pg["text"], "metadata": pg}
    try:
        col = chroma_client.get_collection(f"pdf_{pdf_id}")
    except Exception:
        raise HTTPException(404, "PDF not found")
    result = col.get(ids=[f"{pdf_id}_p{page_num}"], include=["documents", "metadatas"])
    if not result["documents"]:
        raise HTTPException(404, f"Page {page_num} not found")
    return {"page_num": page_num, "text": result["documents"][0], "metadata": result["metadatas"][0]}


@app.get("/api/pdfs")
async def list_pdfs():
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
    update_pdf_record(
        pdf_id, status="vectorized", retrieval_status="vectorized",
        chat_ready=True, pending_pages=0,
    )
    updated = get_pdf_record(pdf_id)
    return {
        "pdf_id": pdf_id, "status": updated["status"],
        "retrieval_status": updated["retrieval_status"],
        "pending_pages": updated["pending_pages"],
        "processed_pages": 0, "indexed_pages": updated["indexed_pages"],
        "chat_ready": updated["chat_ready"],
    }


@app.post("/api/process-pending")
async def process_pending_batch():
    results = []
    for pid in list_pending_pdf_ids():
        results.append(await process_pending_pdf(pid))
    return {"processed": results, "count": len(results)}


@app.delete("/api/pdfs/{pdf_id}")
async def delete_pdf(pdf_id: str):
    deleted_any = False

    try:
        chroma_client.delete_collection(f"pdf_{pdf_id}")
        deleted_any = True
    except Exception:
        pass

    try:
        if get_pdf_record(pdf_id):
            delete_pdf_state(pdf_id)
            deleted_any = True
    except Exception:
        pass

    try:
        p = stored_pdf_path(pdf_id)
        if p.exists():
            p.unlink()
            deleted_any = True
    except Exception:
        pass

    if not deleted_any:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    return {"status": "deleted", "pdf_id": pdf_id}


@app.get("/api/queues")
async def get_queues():
    snapshot = build_queue_snapshot()
    _idle = {
        "running": False,
        "processed": 0,
        "total": 0,
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": 0,
        "status": "idle",
    }
    return {
        **snapshot,
        "runner": {**_idle, "pause_requested": False, "paused": False},
        "index_runner": {**_idle, "finished_pdf_id": "", "finished_filename": ""},
        "stage1_batch_runner": _idle,
        "audit_runner": {**_idle, "flagged": 0},
        "reindex_runner": {**_idle, "fixed": 0},
    }


@app.get("/api/batch-reports")
async def get_batch_reports(limit: int = 8):
    return {"reports": [], "limit": max(1, min(limit, 100))}


# â”€â”€ Startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@app.on_event("startup")
async def startup():
    init_workflow_db()
    log.info("â•â•â• Court File Indexer v5.0  (Local-Only Pipeline) â•â•â•")
    log.info("Text model     : %s", LOCAL_TEXT_MODEL)
    log.info("Vision model   : %s  (assist=%s)", LOCAL_VISION_MODEL, ENABLE_VISION)
    log.info("LLM endpoint   : %s  (timeout=%ss)", LOCAL_LLM_BASE_URL, LOCAL_LLM_TIMEOUT)
    log.info("ChromaDB       : %s", CHROMA_DB_PATH)
    log.info("Tesseract      : lang=%s", TESSERACT_LANG)
    log.info("Parent types   : %s loaded", len(PARENT_DOCUMENT_NAMES))
    log.info("Index exports  : %s", INDEX_EXPORT_PATH)
    log.info("Workflow store : %s (%s)", STORAGE_BACKEND, STORAGE_TARGET)
    log.info("Server ready âœ“")
