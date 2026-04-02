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
import difflib
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
from collections import Counter
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
TOC_STAGE_BUDGET_S = float(os.getenv("TOC_STAGE_BUDGET_S", "90"))
TOC_VISION_TIMEOUT_S = float(os.getenv("TOC_VISION_TIMEOUT_S", "35"))
TOC_TEXT_TIMEOUT_S = float(os.getenv("TOC_TEXT_TIMEOUT_S", "25"))
TOC_MAX_TEXT_LLM_CALLS = int(os.getenv("TOC_MAX_TEXT_LLM_CALLS", "1"))
TOC_MAX_VISION_LLM_CALLS = int(os.getenv("TOC_MAX_VISION_LLM_CALLS", "2"))
TOC_TEXT_FALLBACK_ENABLED = os.getenv("TOC_TEXT_FALLBACK_ENABLED", "true").lower() != "false"
TOC_CIRCUIT_BREAKER_FAILS = int(os.getenv("TOC_CIRCUIT_BREAKER_FAILS", "2"))
TOC_LLM_MAX_RETRIES = int(os.getenv("TOC_LLM_MAX_RETRIES", "1"))
OCR_QUALITY_MIN_FOR_ACCEPT = float(os.getenv("OCR_QUALITY_MIN_FOR_ACCEPT", "0.42"))
INDEX_REVIEW_LOW_ROW_THRESHOLD = int(os.getenv("INDEX_REVIEW_LOW_ROW_THRESHOLD", "1"))
INDEX_ACCEPT_RATIO_MIN = float(os.getenv("INDEX_ACCEPT_RATIO_MIN", "0.80"))
VECTOR_OFFSET_FIX_MAX_SHIFT = int(os.getenv("VECTOR_OFFSET_FIX_MAX_SHIFT", "3"))
ENABLE_WARM_STARTUP = os.getenv("ENABLE_WARM_STARTUP", "true").lower() != "false"
WARM_STARTUP_TIMEOUT_S = float(os.getenv("WARM_STARTUP_TIMEOUT_S", "8"))
GOLDEN_SET_DIR = Path(os.getenv("GOLDEN_SET_DIR", str(Path(__file__).resolve().parent / "golden_set")))

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
    messages: list[dict],
    max_tokens: int = 2000,
    temperature: float = 0.1,
    timeout_s: Optional[float] = None,
) -> str:
    """qwen2.5:14b â€” used ONLY for /api/query chat answers."""
    t_client = _text_client
    temp_http: Optional[httpx.Client] = None
    if timeout_s is not None:
        timeout_s = max(1.0, float(timeout_s))
        if abs(timeout_s - float(LOCAL_LLM_TIMEOUT)) > 0.5:
            temp_http = httpx.Client(timeout=timeout_s)
            t_client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key="ollama", http_client=temp_http)
    try:
        resp = t_client.chat.completions.create(
            model=LOCAL_TEXT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        log.warning("Text LLM call failed: %s", exc)
        return f"[LLM unavailable: {exc}]"
    finally:
        if temp_http is not None:
            temp_http.close()


def call_vision_llm(
    image_b64: str,
    prompt: str,
    max_tokens: int = 2200,
    timeout_s: Optional[float] = None,
) -> str:
    """
    qwen2.5vl:7b â€” used ONLY during ingestion for handwritten / poor-OCR pages.
    Never called during index generation.
    """
    v_client = _vision_client
    temp_http: Optional[httpx.Client] = None
    if timeout_s is not None:
        timeout_s = max(1.0, float(timeout_s))
        if abs(timeout_s - float(LOCAL_LLM_TIMEOUT)) > 0.5:
            temp_http = httpx.Client(timeout=timeout_s)
            v_client = OpenAI(base_url=LOCAL_LLM_BASE_URL, api_key="ollama", http_client=temp_http)
    try:
        resp = v_client.chat.completions.create(
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
    finally:
        if temp_http is not None:
            temp_http.close()


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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ocr_quality_features(text: str) -> dict:
    """
    OCR quality proxy for scanned court pages:
    - table-line detect signal (TOC-style rows)
    - digit consistency signal (page ranges/annexure-like numerics)
    - script-mix signal (Hindi/English balanced extract)
    """
    content = _to_arabic(text or "")
    stats = _text_stats(content)
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    table_rows = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln))
    range_rows = sum(1 for ln in lines if re.search(r"\b\d+\s*(?:-|–|to)\s*\d+\b", ln, flags=re.IGNORECASE))
    digit_token_hits = len(re.findall(r"\b\d+\b", content))
    alpha_chars = len(re.findall(r"[A-Za-z\u0900-\u097F]", content))
    symbol_chars = len(re.findall(r"[^A-Za-z\u0900-\u097F\d\s]", content))
    noisy_symbol_ratio = symbol_chars / max(len(content), 1)
    script_mix = min(stats["dev_ratio"], stats["asc_ratio"]) * 2.0
    digit_density = stats["dig_ratio"]
    line_density = min(stats["lines"] / 18.0, 1.0)
    table_signal = min((table_rows + range_rows) / 6.0, 1.0)
    digit_conf = _clamp01((digit_token_hits / max(stats["words"], 1)) * 2.2)
    alpha_ratio = alpha_chars / max(len(content), 1)
    cleanliness = _clamp01(1.0 - noisy_symbol_ratio * 2.0)
    score = (
        0.20 * line_density
        + 0.22 * table_signal
        + 0.18 * digit_conf
        + 0.16 * script_mix
        + 0.14 * _clamp01(alpha_ratio * 1.3)
        + 0.10 * cleanliness
    )
    return {
        "score": round(_clamp01(score), 3),
        "table_row_hits": table_rows,
        "range_hits": range_rows,
        "digit_confidence": round(digit_conf, 3),
        "script_mix": round(_clamp01(script_mix), 3),
        "noise_ratio": round(noisy_symbol_ratio, 3),
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
        q = _ocr_quality_features(direct)
        return {
            "text":                  re.sub(r"\n{3,}", "\n\n", direct),
            "used_ocr":              False,
            "vision_used":           False,
            "handwriting_suspected": False,
            "extraction_method":     "digital",
            "ocr_quality_score":     q["score"],
            "ocr_quality":           q,
        }

    image    = render_page_image(page, dpi=dpi)
    ocr_text = ocr_page_image(image)

    vision_used           = False
    handwriting_suspected = _needs_vision(direct, ocr_text)

    if handwriting_suspected:
        vision_text = call_vision_llm(
            image_to_jpeg_b64(image),
            _VISION_PROMPT,
            timeout_s=min(float(LOCAL_LLM_TIMEOUT), 22.0),
        )
        if vision_text:
            if _text_stats(vision_text)["chars"] >= max(_text_stats(ocr_text)["chars"], 80):
                ocr_text    = vision_text
                vision_used = True
    q = _ocr_quality_features(ocr_text or "")

    return {
        "text":                  ocr_text or f"[Page {page_num} â€” no readable text]",
        "used_ocr":              True,
        "vision_used":           vision_used,
        "handwriting_suspected": handwriting_suspected,
        "extraction_method":     "vision_ocr" if vision_used else "ocr",
        "ocr_quality_score":     q["score"],
        "ocr_quality":           q,
    }


def extract_pages_from_document(
    doc: fitz.Document,
    page_numbers: list[int],
    total_pages: int,
    dpi: int = 250,
) -> tuple[list[dict], dict]:
    pages_data:               list[dict] = []
    ocr_n = vision_n = hw_n = 0
    quality_sum = 0.0
    quality_count = 0
    low_quality_pages = 0

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
            "ocr_quality_score":     float(pd.get("ocr_quality_score", 0.0)),
            "ocr_quality":           pd.get("ocr_quality", {}),
        })
        q_score = float(pd.get("ocr_quality_score", 0.0))
        quality_sum += q_score
        quality_count += 1
        if q_score < OCR_QUALITY_MIN_FOR_ACCEPT:
            low_quality_pages += 1
        log.info(
            "Page %s/%s â€” %-12s â€” %s chars â€” q=%.3f",
            pn, total_pages, pd["extraction_method"], len(text), q_score,
        )

    return pages_data, {
        "ocr_pages":                   ocr_n,
        "vision_ocr_pages":            vision_n,
        "handwriting_suspected_pages": hw_n,
        "digital_pages":               len(page_numbers) - ocr_n,
        "avg_ocr_quality_score":       round(quality_sum / max(quality_count, 1), 3),
        "low_quality_pages":           low_quality_pages,
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
                "ocr_quality_score":     float(p.get("ocr_quality_score", 0.0)),
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
            "ocr_quality_score":     float(m.get("ocr_quality_score", 0.0) or 0.0),
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

_TOC_NOISE_TITLE_RE = re.compile(
    r"(?:"
    r"\bcourt\s+fees?\b|\bgrand\s+total\b|\breceived\b|\badvocate\b|"
    r"\bapplicant\b|\brespondent\b|\bstate\s+of\b|\bdate\s*[:\-]|\bplace\s*[:\-]|"
    r"\bin\s+the\s+court\b|\bprincipal\s+seat\b|\bmisc\.?\s*criminal\s+case\s+no"
    r")",
    re.IGNORECASE,
)


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


def _cleanup_toc_title(title: str) -> str:
    title = _normalize_toc_line(title or "")
    title = re.sub(r"^[\s\|\[\]\(\)\"'`.,:;]+", "", title)
    title = re.sub(r"[\s\|\[\]\(\)\"'`.,:;]+$", "", title)
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title


def _looks_like_toc_item(item: dict, total_pages: Optional[int] = None) -> bool:
    title = _cleanup_toc_title(str(item.get("title", "")))
    if len(title) < 2:
        return False

    # Drop common non-row header/footer noise that OCR often injects.
    if _TOC_NOISE_TITLE_RE.search(title):
        return False

    # Drop mojibake/noise-heavy lines.
    if "Â" in title or "\ufffd" in title or "°" in title:
        return False

    letters = len(re.findall(r"[A-Za-z\u0900-\u097F]", title))
    digits = len(re.findall(r"\d", title))
    words = re.findall(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F0-9./&-]*", title)
    if letters < 2:
        return False
    if digits > max(letters * 2, 8) and len(words) < 2:
        return False

    # Generic/heading-only lines should not become TOC rows.
    low = title.lower().strip()
    if low in {"index", "table of contents", "contents", "s.no", "sr no"}:
        return False

    fallback_page = max(1, _coerce_int(item.get("pageFrom"), 1))
    pf = _coerce_int(item.get("pageFrom"), fallback_page)
    pt = _coerce_int(item.get("pageTo"), pf)
    if pt < pf:
        pf, pt = pt, pf

    if total_pages is not None:
        if pf > total_pages or pt > total_pages:
            return False
    if pf < 1 or pt < 1:
        return False
    return True


def _sanitize_toc_items(items: list[dict], total_pages: Optional[int] = None) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    page_cap = total_pages if total_pages is not None else 10000

    for item in items:
        if not isinstance(item, dict):
            continue
        title = _cleanup_toc_title(str(item.get("title", "")))
        if not title:
            continue
        row = dict(item)
        row["title"] = title
        pf = max(1, min(_coerce_int(row.get("pageFrom"), 1), page_cap))
        pt = max(pf, min(_coerce_int(row.get("pageTo"), pf), page_cap))
        row["pageFrom"] = pf
        row["pageTo"] = pt
        if not _looks_like_toc_item(row, total_pages=total_pages):
            continue
        key = (title.lower(), pf, pt)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    out.sort(key=lambda x: (_coerce_int(x.get("pageFrom"), 1), _coerce_int(x.get("pageTo"), 1), x.get("title", "")))

    # OCR often emits multiple fake rows on the last page number.
    # Trim those only when they dominate and look weak.
    if total_pages is not None and len(out) >= 4:
        starts = [_coerce_int(x.get("pageFrom"), 1) for x in out]
        top_page, top_count = Counter(starts).most_common(1)[0]
        if top_page >= max(3, total_pages - 1) and top_count >= max(2, int(math.ceil(len(out) * 0.5))):
            trimmed: list[dict] = []
            for row in out:
                pf = _coerce_int(row.get("pageFrom"), 1)
                raw_source = str(row.get("source", "")).lower()
                serial = str(row.get("serialNo", "")).strip()
                weak_tail = (
                    pf == top_page
                    and not serial
                    and (raw_source.startswith("toc-stitch") or raw_source.startswith("toc-regex"))
                )
                if weak_tail:
                    continue
                trimmed.append(row)
            if len(trimmed) >= 2:
                out = trimmed

    return out


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
    candidates: list[str] = [payload]
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidates.insert(0, m.group(1).strip())
    for start_char, end_char in (("[", "]"), ("{", "}")):
        si = payload.find(start_char)
        ei = payload.rfind(end_char)
        if si != -1 and ei != -1 and ei > si:
            candidates.append(payload[si : ei + 1].strip())

    parsed = None
    decoder = json.JSONDecoder()
    for cand in candidates:
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
            break
        except Exception:
            pass
        first_struct = min([i for i in (cand.find("["), cand.find("{")) if i >= 0], default=-1)
        if first_struct >= 0:
            try:
                parsed, _ = decoder.raw_decode(cand[first_struct:])
                break
            except Exception:
                pass
    if parsed is None:
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


def _toc_rows_from_local_llm(
    text: str, fallback_page: int = 1, timeout_s: Optional[float] = None
) -> list[dict]:
    prompt = (
        "Extract only table-of-contents/index rows from this OCR text. "
        "Return strict JSON array only, no prose. "
        "Schema for each row: "
        '{"serialNo":"", "title":"", "annexure":"", "pageFrom":1, "pageTo":1}. '
        "If not found, return [].\n\n"
        f"Text:\n{text[:9000]}"
    )
    raw = ""
    for _ in range(max(1, TOC_LLM_MAX_RETRIES)):
        raw = call_text_llm(
            [{"role": "user", "content": prompt}],
            max_tokens=1400,
            temperature=0.0,
            timeout_s=timeout_s,
        )
        if raw and not raw.startswith("[LLM unavailable:"):
            break
    rows = _parse_json_list(raw)
    if not rows:
        return []
    for row in rows:
        row["pageFrom"] = max(1, _coerce_int(row.get("pageFrom"), fallback_page))
        row["pageTo"] = max(row["pageFrom"], _coerce_int(row.get("pageTo"), row["pageFrom"]))
    return rows


def _toc_rows_from_vision_llm(
    image: Image.Image, fallback_page: int = 1, timeout_s: Optional[float] = None
) -> list[dict]:
    if not ENABLE_VISION:
        return []
    prompt = (
        "You are reading a scanned Indian court-file Table of Contents / Index page. "
        "The page may be in English, Hindi (Devanagari), or mixed text, and may include handwritten entries.\n\n"
        "Task:\n"
        "- Decide whether this page is a real TOC / index / सूची / विषय सूची / अनुक्रमणिका page.\n"
        "- If it is, extract every readable table row from the page image.\n"
        "- Preserve the row description exactly as written.\n"
        "- Convert Hindi digits to Arabic numerals in pageFrom/pageTo.\n"
        "- If a row spans one page, set pageFrom and pageTo to the same value.\n"
        "- If a row title continues on the next line in the same row, combine it.\n"
        "- Return [] if this page is not actually a TOC.\n\n"
        "Return strict JSON array only (no prose): "
        '[{"serialNo":"", "title":"", "annexure":"", "pageFrom":1, "pageTo":1}].'
    )
    raw = ""
    for _ in range(max(1, TOC_LLM_MAX_RETRIES)):
        raw = call_vision_llm(
            image_to_jpeg_b64(image), prompt, max_tokens=1400, timeout_s=timeout_s
        )
        if raw:
            break
    rows = _parse_json_list(raw)
    if not rows:
        return []
    for row in rows:
        row["pageFrom"] = max(1, _coerce_int(row.get("pageFrom"), fallback_page))
        row["pageTo"] = max(row["pageFrom"], _coerce_int(row.get("pageTo"), row["pageFrom"]))
    return rows


def extract_toc_rows_vision_first(
    pdf_id: str,
    candidates: list[dict],
    total_pages: int,
    max_pages: int = 2,
    deadline_ts: Optional[float] = None,
    stage_stats: Optional[dict] = None,
) -> tuple[list[dict], list[int]]:
    """
    Borrowed from older high-accuracy behavior:
    try image-based table extraction on top TOC candidates before regex-only parsing.
    """
    if not ENABLE_VISION or not candidates:
        return [], []

    collected: list[dict] = []
    used_pages: list[int] = []
    for page in candidates[: max(1, max_pages)]:
        if deadline_ts is not None and time.perf_counter() >= deadline_ts:
            log.warning("TOC stage budget reached before vision-first completed")
            break
        page_num = int(page["page_num"])
        img = _load_pdf_page_image(pdf_id, page_num, dpi=300)
        if img is None:
            continue
        call_timeout = None
        if deadline_ts is not None:
            remaining = max(0.5, deadline_ts - time.perf_counter())
            call_timeout = min(TOC_VISION_TIMEOUT_S, remaining)
        t0 = time.perf_counter()
        rows = _toc_rows_from_vision_llm(img, fallback_page=page_num, timeout_s=call_timeout)
        dt = time.perf_counter() - t0
        if stage_stats is not None:
            stage_stats["toc_vision_calls"] = stage_stats.get("toc_vision_calls", 0) + 1
            stage_stats["toc_vision_time_s"] = stage_stats.get("toc_vision_time_s", 0.0) + dt
            if not rows:
                stage_stats["toc_vision_failures"] = stage_stats.get("toc_vision_failures", 0) + 1
        rows = _sanitize_toc_items(rows, total_pages=total_pages)
        if rows:
            collected.extend(rows)
            collected = _sanitize_toc_items(collected, total_pages=total_pages)
            used_pages.append(page_num)
            log.info("Vision-first TOC rows p=%s -> %s", page_num, len(rows))
            if _is_good_toc_extraction(collected, total_pages):
                break

    return collected, used_pages


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
    next_offsets: tuple[int, ...] = (1,),
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


def extract_toc_rows_with_fallback(
    pdf_id: str,
    page: dict,
    allow_text_llm: bool = False,
    allow_vision_llm: bool = False,
    total_pages: Optional[int] = None,
    deadline_ts: Optional[float] = None,
    stage_stats: Optional[dict] = None,
) -> tuple[list[dict], str]:
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

    current_rows = _sanitize_toc_items(best_rows, total_pages=total_pages)
    should_try_text_llm = (
        TOC_TEXT_FALLBACK_ENABLED
        and
        allow_text_llm
        and len(current_rows) == 0
        and bool(_STRUCTURAL_RE.search(best_text or base_text or ""))
        and (
            (not ENABLE_VISION)
            or (stage_stats is not None and int(stage_stats.get("toc_vision_failures", 0)) > 0)
        )
        and (stage_stats is None or int(stage_stats.get("toc_text_calls", 0)) < max(0, TOC_MAX_TEXT_LLM_CALLS))
    )
    if should_try_text_llm:
        if deadline_ts is not None and time.perf_counter() >= deadline_ts:
            log.warning("TOC stage budget reached before text fallback on p=%s", page_num)
        else:
            call_timeout = None
            if deadline_ts is not None:
                remaining = max(0.5, deadline_ts - time.perf_counter())
                call_timeout = min(TOC_TEXT_TIMEOUT_S, remaining)
            t0 = time.perf_counter()
            llm_rows = _toc_rows_from_local_llm(best_text, fallback_page=page_num, timeout_s=call_timeout)
            dt = time.perf_counter() - t0
            if stage_stats is not None:
                stage_stats["toc_text_calls"] = stage_stats.get("toc_text_calls", 0) + 1
                stage_stats["toc_text_time_s"] = stage_stats.get("toc_text_time_s", 0.0) + dt
                if not llm_rows:
                    stage_stats["toc_text_failures"] = stage_stats.get("toc_text_failures", 0) + 1
            if len(llm_rows) > len(best_rows):
                best_rows, best_method = llm_rows, "toc-text-llm"

    current_rows = _sanitize_toc_items(best_rows, total_pages=total_pages)
    should_try_vision_llm = (
        ENABLE_VISION
        and allow_vision_llm
        and len(current_rows) == 0
        and (stage_stats is None or int(stage_stats.get("toc_vision_calls", 0)) < max(0, TOC_MAX_VISION_LLM_CALLS))
    )
    if should_try_vision_llm:
        if deadline_ts is not None and time.perf_counter() >= deadline_ts:
            log.warning("TOC stage budget reached before vision fallback on p=%s", page_num)
        else:
            img = _load_pdf_page_image(pdf_id, page_num, dpi=300)
            if img is not None:
                call_timeout = None
                if deadline_ts is not None:
                    remaining = max(0.5, deadline_ts - time.perf_counter())
                    call_timeout = min(TOC_VISION_TIMEOUT_S, remaining)
                t0 = time.perf_counter()
                vision_rows = _toc_rows_from_vision_llm(img, fallback_page=page_num, timeout_s=call_timeout)
                dt = time.perf_counter() - t0
                if stage_stats is not None:
                    stage_stats["toc_vision_calls"] = stage_stats.get("toc_vision_calls", 0) + 1
                    stage_stats["toc_vision_time_s"] = stage_stats.get("toc_vision_time_s", 0.0) + dt
                    if not vision_rows:
                        stage_stats["toc_vision_failures"] = stage_stats.get("toc_vision_failures", 0) + 1
                if len(vision_rows) > len(best_rows):
                    best_rows, best_method = vision_rows, "toc-vision-llm"

    return _sanitize_toc_items(best_rows, total_pages=total_pages), best_method


def _dedupe_toc_items(items: list[dict]) -> list[dict]:
    return _sanitize_toc_items(items, total_pages=None)


def _is_good_toc_extraction(items: list[dict], total_pages: int) -> bool:
    deduped = _sanitize_toc_items(items, total_pages=total_pages)
    if len(deduped) < 3:
        return False
    starts = [_coerce_int(item.get("pageFrom"), 1) for item in deduped]
    distinct_pages = sorted(set(starts))
    if len(distinct_pages) < 3:
        return False
    if Counter(starts).most_common(1)[0][1] > max(2, int(math.ceil(len(starts) * 0.6))):
        return False
    max_page = max((_coerce_int(item.get("pageTo"), _coerce_int(item.get("pageFrom"), 1)) for item in deduped), default=1)
    min_page = min(starts)
    if (max_page - min_page) < 2:
        return False
    return max_page >= min(total_pages, 4)


def build_toc_processing_order(
    top_candidates: list[dict],
    expanded_candidates: list[dict],
    max_pages: int = 6,
) -> list[dict]:
    ordered: list[dict] = []
    seen: set[int] = set()
    for page in top_candidates + expanded_candidates:
        page_num = int(page["page_num"])
        if page_num in seen:
            continue
        seen.add(page_num)
        ordered.append(page)
        if len(ordered) >= max_pages:
            break
    return ordered


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


def _normalize_toc_index_items(
    items: list[dict], indexed_start: int, indexed_end: int, default_source: str
) -> list[dict]:
    """
    Backup-style normalization before range fill:
      - sanitize noisy rows
      - clamp page bounds
      - strict dedupe by (title, pageFrom, pageTo)
    """
    sanitized = _sanitize_toc_items(items, total_pages=indexed_end)
    out: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for item in sanitized:
        title = _cleanup_toc_title(str(item.get("title", "")))
        if not title:
            continue
        pf = _coerce_int(item.get("pageFrom"), indexed_start)
        pt = _coerce_int(item.get("pageTo"), pf)
        pf = max(indexed_start, min(pf, indexed_end))
        pt = max(pf, min(pt, indexed_end))
        key = (title.lower(), pf, pt)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            **item,
            "title": title,
            "pageFrom": pf,
            "pageTo": pt,
            "source": str(item.get("source", default_source) or default_source),
        })
    out.sort(key=lambda x: (x["pageFrom"], x["pageTo"], x["title"]))
    return out


def _toc_title_key(title: str) -> str:
    t = _normalize_label(_cleanup_toc_title(title or ""))
    t = re.sub(r"\b(?:copy|certified|dated|dt|order|application|memo|index)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _merge_near_duplicate_toc_items(items: list[dict]) -> list[dict]:
    """
    Merge near-duplicate titles that share the same page range.
    Keeps the longer / cleaner title variant.
    """
    by_range: dict[tuple[int, int], list[dict]] = {}
    for item in items:
        pf = _coerce_int(item.get("pageFrom"), 1)
        pt = _coerce_int(item.get("pageTo"), pf)
        by_range.setdefault((pf, pt), []).append(item)

    merged: list[dict] = []
    for (pf, pt), group in by_range.items():
        kept: list[dict] = []
        for item in group:
            title = str(item.get("title", "")).strip()
            key = _toc_title_key(title)
            if not key:
                continue
            dup_idx = None
            for i, ex in enumerate(kept):
                ex_key = _toc_title_key(str(ex.get("title", "")))
                ratio = difflib.SequenceMatcher(None, key, ex_key).ratio()
                if ratio >= 0.86 or key in ex_key or ex_key in key:
                    dup_idx = i
                    break
            if dup_idx is None:
                kept.append(item)
            else:
                old_title = str(kept[dup_idx].get("title", ""))
                if len(title) > len(old_title):
                    kept[dup_idx] = item
        merged.extend(kept)
    merged.sort(key=lambda x: (_coerce_int(x.get("pageFrom"), 1), _coerce_int(x.get("pageTo"), 1), str(x.get("title", ""))))
    return merged


def _estimate_toc_rows_from_page_text(text: str) -> int:
    text = _to_arabic(text or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    serial_lines = sum(1 for ln in lines if re.match(r"^\s*[०-९\d]{1,3}[\.\)\/:-]?\s+", ln))
    table_lines = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln))
    return max(serial_lines, table_lines)


def evaluate_toc_structure(
    toc_items: list[dict],
    total_pages: int,
    toc_hint_text: str = "",
) -> dict:
    items = _sanitize_toc_items(toc_items, total_pages=total_pages)
    items = _merge_near_duplicate_toc_items(items)
    count = len(items)
    starts = [_coerce_int(it.get("pageFrom"), 1) for it in items]
    page_majority_ratio = 0.0
    if starts:
        page_majority_ratio = Counter(starts).most_common(1)[0][1] / max(1, len(starts))

    valid_rows = 0
    invalid_rows = 0
    for it in items:
        title = str(it.get("title", "")).strip()
        pf = _coerce_int(it.get("pageFrom"), 1)
        pt = _coerce_int(it.get("pageTo"), pf)
        ok = bool(title) and 1 <= pf <= total_pages and 1 <= pt <= total_pages and pf <= pt
        if ok:
            valid_rows += 1
        else:
            invalid_rows += 1

    expected_rows = _estimate_toc_rows_from_page_text(toc_hint_text) if toc_hint_text else 0
    noisy_count_mismatch = expected_rows >= 4 and count >= max(8, expected_rows * 2)
    dominant_page_anomaly = page_majority_ratio >= 0.5
    weak_continuity = False
    if count >= 4:
        sorted_items = sorted(items, key=lambda x: (_coerce_int(x.get("pageFrom"), 1), _coerce_int(x.get("pageTo"), 1)))
        big_jumps = 0
        for i in range(len(sorted_items) - 1):
            a = _coerce_int(sorted_items[i].get("pageFrom"), 1)
            b = _coerce_int(sorted_items[i + 1].get("pageFrom"), 1)
            if b - a > max(8, int(total_pages * 0.55)):
                big_jumps += 1
        weak_continuity = big_jumps >= 2

    reasons: list[str] = []
    if count < 2:
        reasons.append("row_count_lt_2")
    if invalid_rows > 0:
        reasons.append("invalid_fields_or_ranges")
    if noisy_count_mismatch:
        reasons.append("row_count_noisy_vs_toc")
    if dominant_page_anomaly:
        reasons.append("dominant_same_page_from")
    if weak_continuity:
        reasons.append("range_continuity_weak")

    if count < 2 or invalid_rows > 0:
        decision = "REJECT_TOC_USE_FALLBACK"
    elif dominant_page_anomaly or noisy_count_mismatch:
        decision = "REJECT_TOC_USE_FALLBACK"
    elif weak_continuity:
        decision = "REVIEW"
    else:
        decision = "ACCEPT"

    return {
        "decision": decision,
        "reasons": reasons,
        "row_count": count,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "expected_rows_hint": expected_rows,
        "dominant_page_ratio": round(page_majority_ratio, 3),
        "items": items,
    }


def apply_row_confidence_checks(
    items: list[dict], total_pages: int, far_gap_threshold: int = 6
) -> dict:
    out: list[dict] = []
    high = medium = low = 0
    for item in items:
        pf = _coerce_int(item.get("pageFrom"), 1)
        pt = _coerce_int(item.get("pageTo"), pf)
        title = str(item.get("title", "")).strip()
        matched = [int(p) for p in (item.get("matchedPages") or []) if isinstance(p, int) or str(p).isdigit()]
        top_match = matched[0] if matched else None
        gap = abs(top_match - pf) if top_match is not None else 999
        range_ok = 1 <= pf <= total_pages and 1 <= pt <= total_pages and pf <= pt
        noisy = _is_noisy_index_title(title)
        v_status = str(item.get("verificationStatus", ""))

        if not range_ok or noisy:
            level = "low"
        elif v_status == "verified" and gap <= 2:
            level = "high"
        elif top_match is not None and gap > far_gap_threshold:
            level = "low"
        else:
            level = "medium"

        if level == "high":
            high += 1
        elif level == "medium":
            medium += 1
        else:
            low += 1

        out.append({
            **item,
            "rowConfidence": level,
            "vectorGap": gap if top_match is not None else None,
            "topMatchedPage": top_match,
        })

    total = max(1, len(out))
    accept_like_ratio = (high + medium) / total
    if len(out) == 0:
        decision = "REJECT_TOC_USE_FALLBACK"
    elif low > 0:
        decision = "REVIEW"
    elif accept_like_ratio >= 0.8:
        decision = "ACCEPT"
    else:
        decision = "REVIEW"

    return {
        "decision": decision,
        "high": high,
        "medium": medium,
        "low": low,
        "accept_like_ratio": round(accept_like_ratio, 3),
        "items": out,
    }


def build_toc_ranges_from_items(
    items: list[dict], indexed_start: int, range_end: int, default_source: str
) -> list[dict]:
    """
    Normalize â†’ sort â†’ deduplicate â†’ forward-fill page ranges.
    Each item's pageTo = next item's pageFrom âˆ’ 1  (when unambiguous).
    """
    out: list[dict] = []
    normalized = _normalize_toc_index_items(items, indexed_start, range_end, default_source)

    for item in normalized:
        title = str(item.get("title", "")).strip()
        pf = _coerce_int(item.get("pageFrom"), indexed_start)
        pt = _coerce_int(item.get("pageTo"), pf)
        raw_source = str(item.get("source", default_source) or default_source)
        ui_source = "toc" if raw_source.startswith("toc") else default_source
        out.append({
            "title":         title,
            "displayTitle":  str(item.get("displayTitle")  or title).strip(),
            "originalTitle": str(item.get("originalTitle") or title).strip(),
            "pageFrom":      pf,
            "pageTo":        pt,
            "tocPageFrom":   pf,
            "tocPageTo":     pt,
            "pdfPageFrom":   pf,
            "pdfPageTo":     pt,
            "source":        ui_source,
            "rawSource":     raw_source,
            "serialNo":      str(item.get("serialNo",  "")),
            "annexure":      str(item.get("annexure",  "")),
            "courtFee":      str(item.get("courtFee",  "")),
        })

    out.sort(key=lambda x: (x["pageFrom"], x["pageTo"], x["title"]))

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
    offset_votes: list[int] = []
    for idx, item in enumerate(index_items):
        title = (
            item.get("displayTitle") or item.get("originalTitle") or item.get("title") or ""
        ).strip()

        if not title:
            verified.append({
                **item,
                "tocPageFrom":           item.get("tocPageFrom", item.get("pageFrom", 1)),
                "tocPageTo":             item.get("tocPageTo", item.get("pageTo", 1)),
                "pdfPageFrom":           item.get("pdfPageFrom", item.get("pageFrom", 1)),
                "pdfPageTo":             item.get("pdfPageTo", item.get("pageTo", 1)),
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
            if status in {"verified", "weak_match"}:
                offset_votes.append(int(nearest - toc_from))

        if next_from is not None and verified_from < next_from:
            verified_to = max(verified_from, next_from - 1)
        else:
            verified_to = max(verified_from, toc_to)

        verified.append({
            **item,
            "pageFrom":               toc_from,
            "pageTo":                 toc_to,
            "tocPageFrom":            item.get("tocPageFrom", toc_from),
            "tocPageTo":              item.get("tocPageTo", toc_to),
            "pdfPageFrom":            verified_from,
            "pdfPageTo":              verified_to,
            "verifiedPageFrom":       verified_from,
            "verifiedPageTo":         verified_to,
            "verificationStatus":     status,
            "verificationConfidence": confidence,
            "matchedPages":           top_hits[:5],
        })

    # Post-verify offset correction:
    # if most rows indicate the same small page offset, shift all toc ranges consistently.
    if len(verified) >= 3 and offset_votes:
        filtered = [o for o in offset_votes if abs(o) <= max(1, VECTOR_OFFSET_FIX_MAX_SHIFT)]
        if filtered:
            common_offset, common_count = Counter(filtered).most_common(1)[0]
            if common_offset != 0 and common_count >= max(2, int(math.ceil(len(verified) * 0.6))):
                corrected: list[dict] = []
                for item in verified:
                    toc_from = int(item.get("pageFrom", 1))
                    toc_to = int(item.get("pageTo", toc_from))
                    new_from = max(1, toc_from + common_offset)
                    new_to = max(new_from, toc_to + common_offset)
                    corrected.append({
                        **item,
                        "pdfPageFrom": new_from,
                        "pdfPageTo": new_to,
                        "verifiedPageFrom": new_from,
                        "verifiedPageTo": new_to,
                        "verificationStatus": "offset_corrected",
                    })
                verified = corrected

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


_FALLBACK_TITLE_SKIP_RE = re.compile(
    r"^(?:"
    r"in\s+the\b|before\s+the\b|high\s+court\b|district\s+court\b|"
    r"applicant\b|respondent\b|petitioner\b|versus\b|vs\.?\b|"
    r"index\b|table\s+of\s+contents\b|sr\.?\s*no\b|particulars\b|annexure\b|"
    r"page\b|dated\b|advocate\b|received\b|jabalpur\b|memo\s+of\s+appearance\b"
    r")",
    re.IGNORECASE,
)


def _pick_fallback_page_title(page: dict, doc_type: str) -> str:
    text = _to_arabic(page.get("text", "") or "")
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" .:-")
        if len(line) < 5:
            continue
        if _FALLBACK_TITLE_SKIP_RE.search(line):
            continue
        if re.fullmatch(r"[\d\s./-]+", line):
            continue
        return line[:140]
    return doc_type or f"Page {int(page.get('page_num', 1))}"


def build_classification_fallback_index(all_pages: list[dict]) -> list[dict]:
    """
    Fallback when no reliable TOC/index page is extracted.
    Build contiguous page spans from per-page document classification.
    """
    if not all_pages:
        return []

    provisional: list[dict] = []
    for page in sorted(all_pages, key=lambda x: x["page_num"]):
        page_num = int(page["page_num"])
        preview = (page.get("text", "") or "").strip()[:1400]
        doc_type = _direct_alias(preview, preview)
        if not doc_type and PARENT_DOCUMENT_NAMES:
            scored = _score_parent_docs(preview[:240], preview)
            non_gen = [(s, n) for s, n in scored if n.lower() not in GENERIC_PARENT_NAMES]
            pool = non_gen or scored
            doc_type = pool[0][1] if pool else "Other"
        doc_type = doc_type or "Other"
        title = _pick_fallback_page_title(page, doc_type)
        provisional.append({
            "title": title if doc_type == "Other" else doc_type,
            "displayTitle": title,
            "originalTitle": title,
            "pageFrom": page_num,
            "pageTo": page_num,
            "tocPageFrom": "",
            "tocPageTo": "",
            "pdfPageFrom": page_num,
            "pdfPageTo": page_num,
            "verifiedPageFrom": page_num,
            "verifiedPageTo": page_num,
            "source": "auto",
            "rawSource": "classification-fallback",
            "verificationStatus": "classification_fallback",
            "verificationConfidence": 0.45,
            "matchedPages": [page_num],
            "documentType": doc_type,
            "serialNo": "",
            "annexure": "",
            "courtFee": "",
        })

    return _merge_adjacent(provisional)


_TITLE_NOISE_RE = re.compile(r"(?:\uFFFD|Â|[^\w\s./,&():'-]{3,})", re.UNICODE)


def _is_noisy_index_title(title: str) -> bool:
    t = (title or "").strip()
    if len(t) < 4:
        return True
    if _TITLE_NOISE_RE.search(t):
        return True
    letters = len(re.findall(r"[A-Za-z\u0900-\u097F]", t))
    digits = len(re.findall(r"\d", t))
    if letters == 0:
        return True
    if digits > max(letters * 2, 10):
        return True
    return False


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

        clean_title = doc_type if _is_noisy_index_title(title) and doc_type and doc_type != "Other" else title
        result.append({
            **item,
            "title":         clean_title,
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
            prev["pdfPageTo"]      = item.get("pdfPageTo", item["pageTo"])
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


def _normalize_eval_title(text: str) -> str:
    cleaned = _cleanup_toc_title(text or "")
    cleaned = _normalize_label(cleaned)
    cleaned = re.sub(r"\b(?:copy|certified|dated|dt)\b", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _normalize_eval_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows or []:
        title = _normalize_eval_title(
            str(row.get("displayTitle") or row.get("originalTitle") or row.get("title") or "")
        )
        pf = _coerce_int(row.get("pageFrom"), 1)
        pt = _coerce_int(row.get("pageTo"), pf)
        if not title or pf <= 0 or pt < pf:
            continue
        key = (title, pf, pt)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"title": title, "pageFrom": pf, "pageTo": pt})
    normalized.sort(key=lambda x: (x["pageFrom"], x["pageTo"], x["title"]))
    return normalized


def _evaluate_index_accuracy(pred_rows: list[dict], exp_rows: list[dict]) -> dict:
    pred_norm = _normalize_eval_rows(pred_rows)
    exp_norm = _normalize_eval_rows(exp_rows)
    pred_set = {(r["title"], r["pageFrom"], r["pageTo"]) for r in pred_norm}
    exp_set = {(r["title"], r["pageFrom"], r["pageTo"]) for r in exp_norm}
    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    exact = pred_set == exp_set and len(exp_set) > 0
    return {
        "expected_rows": len(exp_norm),
        "predicted_rows": len(pred_norm),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_match": bool(exact),
    }


def _load_golden_specs(limit: int = 50) -> list[dict]:
    specs: list[dict] = []
    if not GOLDEN_SET_DIR.exists():
        return specs
    files = sorted(GOLDEN_SET_DIR.glob("*.json"))[: max(1, min(limit, 200))]
    for fp in files:
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
            pdf_id = str(payload.get("pdf_id") or "").strip()
            expected = payload.get("expected_index") or payload.get("expected") or []
            if not pdf_id or not isinstance(expected, list):
                continue
            specs.append({"pdf_id": pdf_id, "expected_index": expected, "name": fp.name})
        except Exception as exc:
            log.warning("Golden spec parse failed (%s): %s", fp.name, exc)
    return specs


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
        "toc_stage_budget_s": TOC_STAGE_BUDGET_S,
        "toc_vision_timeout_s": TOC_VISION_TIMEOUT_S,
        "toc_text_timeout_s": TOC_TEXT_TIMEOUT_S,
        "toc_text_fallback_enabled": TOC_TEXT_FALLBACK_ENABLED,
        "toc_max_text_calls": TOC_MAX_TEXT_LLM_CALLS,
        "toc_max_vision_calls": TOC_MAX_VISION_LLM_CALLS,
        "toc_circuit_breaker_fails": TOC_CIRCUIT_BREAKER_FAILS,
        "ocr_quality_min_for_accept": OCR_QUALITY_MIN_FOR_ACCEPT,
        "warm_startup": ENABLE_WARM_STARTUP,
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
        ingest_t0 = time.perf_counter()
        pages_data, stats = extract_pages_from_document(
            doc, list(range(1, total_pages + 1)), total_pages, dpi=250
        )
        ocr_elapsed = time.perf_counter() - ingest_t0
        log.info("Ingest OCR/extraction time: %.2fs for %s pages", ocr_elapsed, total_pages)
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
    req_t0 = time.perf_counter()
    stage_timings: dict[str, float] = {}
    toc_stage_stats: dict[str, float] = {
        "toc_vision_calls": 0,
        "toc_vision_time_s": 0.0,
        "toc_text_calls": 0,
        "toc_text_time_s": 0.0,
        "toc_vision_failures": 0,
        "toc_text_failures": 0,
    }

    record = get_pdf_record(req.pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {req.pdf_id} not found. Please ingest first.")

    all_pages = load_all_pages_for_pdf(req.pdf_id)
    if not all_pages:
        raise HTTPException(404, f"No page data for PDF {req.pdf_id}. Please ingest first.")

    all_pages.sort(key=lambda x: x["page_num"])
    total_pages = record["total_pages"]
    toc_window_end = min(max(25, min(40, total_pages)), total_pages)
    toc_window = [p for p in all_pages if 1 <= p["page_num"] <= toc_window_end]

    # Step 1 - hybrid TOC ranking in pages 1..25
    t0 = time.perf_counter()
    top_candidates = detect_toc_candidate_pages(toc_window, max_candidates=4)
    candidate_nums = [int(p["page_num"]) for p in top_candidates]
    expanded_candidates = expand_toc_candidate_pages(top_candidates[:2], all_pages)
    expanded_candidate_nums = [int(p["page_num"]) for p in expanded_candidates]
    toc_processing_order = build_toc_processing_order(top_candidates, expanded_candidates, max_pages=6)
    toc_processing_nums = [int(p["page_num"]) for p in toc_processing_order]
    stage_timings["toc_detect_s"] = round(time.perf_counter() - t0, 3)
    log.info("TOC top candidates: %s", candidate_nums)
    log.info("TOC expanded candidates: %s", expanded_candidate_nums)
    log.info("TOC processing order: %s", toc_processing_nums)

    # Step 2 - parse TOC rows with fast stop and limited fallback
    t0 = time.perf_counter()
    toc_deadline_ts = t0 + max(5.0, TOC_STAGE_BUDGET_S)
    raw_items: list[dict] = []
    vision_first_pages_used: list[int] = []
    if ENABLE_VISION and top_candidates:
        vision_seed_items, vision_first_pages_used = extract_toc_rows_vision_first(
            req.pdf_id,
            top_candidates,
            total_pages=total_pages,
            max_pages=2,
            deadline_ts=toc_deadline_ts,
            stage_stats=toc_stage_stats,
        )
        if vision_seed_items:
            raw_items.extend(vision_seed_items)
            raw_items = _sanitize_toc_items(raw_items, total_pages=total_pages)
            log.info(
                "Vision-first TOC seed rows -> %s (pages=%s)",
                len(vision_seed_items),
                vision_first_pages_used,
            )

    vision_seed_usable = len(raw_items) >= 2
    if _is_good_toc_extraction(raw_items, total_pages):
        log.info("Early stop after strong vision-first TOC extraction")
        vision_seed_usable = True

    llm_allowed_pages = {int(p["page_num"]) for p in top_candidates[:2]}
    # Strong-path policy:
    # - If vision-first already produced usable rows, do not mix non-vision rows.
    # - Run OCR/text fallback only when zero usable rows are available.
    if not vision_seed_usable and len(raw_items) == 0:
        for idx, page in enumerate(toc_processing_order):
            if time.perf_counter() >= toc_deadline_ts:
                log.warning("TOC stage budget exhausted; proceeding with current rows")
                break
            if _is_good_toc_extraction(raw_items, total_pages):
                break
            page_num = int(page["page_num"])
            if page_num in vision_first_pages_used:
                continue
            parsed, method = extract_toc_rows_with_fallback(
                req.pdf_id,
                page,
                allow_text_llm=page_num in llm_allowed_pages,
                allow_vision_llm=page_num in llm_allowed_pages and idx == 0,
                total_pages=total_pages,
                deadline_ts=toc_deadline_ts,
                stage_stats=toc_stage_stats,
            )
            if parsed:
                log.info("TOC rows p=%s -> %s via %s", page_num, len(parsed), method)
                raw_items.extend(parsed)
                raw_items = _sanitize_toc_items(raw_items, total_pages=total_pages)
                if idx == 0 and _is_good_toc_extraction(parsed, total_pages):
                    log.info("Early stop after strong TOC extraction on page %s", page_num)
                    break
                if _is_good_toc_extraction(raw_items, total_pages):
                    log.info("Early stop after accumulating strong TOC evidence by page %s", page_num)
                    break
    elif vision_seed_usable:
        log.info("Using vision-first TOC rows only; skipping non-vision fallback merge")

    raw_items = _sanitize_toc_items(raw_items, total_pages=total_pages)
    stage_timings["toc_extract_s"] = round(time.perf_counter() - t0, 3)
    toc_hint_text = (top_candidates[0].get("text", "") if top_candidates else "") or ""
    toc_quality = evaluate_toc_structure(raw_items, total_pages=total_pages, toc_hint_text=toc_hint_text)
    raw_items = toc_quality["items"]
    llm_failures = int(toc_stage_stats.get("toc_text_failures", 0)) + int(toc_stage_stats.get("toc_vision_failures", 0))
    if llm_failures >= max(1, TOC_CIRCUIT_BREAKER_FAILS) and len(raw_items) < 2:
        toc_quality["decision"] = "REJECT_TOC_USE_FALLBACK"
        existing_reasons = list(toc_quality.get("reasons") or [])
        if "circuit_breaker_llm_failures" not in existing_reasons:
            existing_reasons.append("circuit_breaker_llm_failures")
        toc_quality["reasons"] = existing_reasons
    log.info("TOC structural quality: %s", {k: v for k, v in toc_quality.items() if k != "items"})

    # Step 3 â€” build page ranges
    index_items: list[dict] = []
    index_source = "toc"
    if toc_quality.get("decision") != "REJECT_TOC_USE_FALLBACK" and len(raw_items) >= 2:
        index_items = build_toc_ranges_from_items(
            raw_items, indexed_start=1, range_end=total_pages, default_source="toc"
        )

    if not index_items:
        log.info(
            "No reliable TOC rows found for %s. Falling back to page-span classification.",
            req.pdf_id,
        )
        index_items = build_classification_fallback_index(all_pages)
        index_source = "classification-fallback"

    if not index_items:
        raise HTTPException(422, detail="Unable to build index from TOC or fallback classification.")

    # Step 4 â€” vector verification
    t0 = time.perf_counter()
    verified = (
        verify_index_items_with_vectors(req.pdf_id, index_items, all_pages)
        if index_source == "toc"
        else index_items
    )
    stage_timings["verify_s"] = round(time.perf_counter() - t0, 3)
    row_quality = (
        apply_row_confidence_checks(verified, total_pages=total_pages)
        if index_source == "toc"
        else {"decision": "ACCEPT", "high": 0, "medium": len(verified), "low": 0, "accept_like_ratio": 1.0, "items": verified}
    )
    verified = row_quality["items"]
    log.info("TOC row quality: %s", {k: v for k, v in row_quality.items() if k != "items"})

    # Step 5 â€” classify document types
    t0 = time.perf_counter()
    classified = (
        classify_index_items(verified, all_pages)
        if index_source == "toc"
        else verified
    )
    stage_timings["classify_s"] = round(time.perf_counter() - t0, 3)

    # Step 6 â€” persist
    final_decision = "ACCEPT"
    review_reason = ""
    if index_source == "toc":
        if toc_quality.get("decision") == "REJECT_TOC_USE_FALLBACK":
            final_decision = "REJECT_TOC_USE_FALLBACK"
            review_reason = "structural_failure"
        else:
            low_rows = int(row_quality.get("low", 0))
            accept_like_ratio = float(row_quality.get("accept_like_ratio", 1.0))
            toc_reasons = list(toc_quality.get("reasons") or [])
            critical_toc_review = any(
                reason in {"range_continuity_weak", "row_count_noisy_vs_toc"}
                for reason in toc_reasons
            )
            if low_rows >= max(1, INDEX_REVIEW_LOW_ROW_THRESHOLD):
                final_decision = "REJECT_TOC_USE_FALLBACK"
                review_reason = "low_confidence_rows"
            elif len(verified) >= 3 and accept_like_ratio < INDEX_ACCEPT_RATIO_MIN:
                final_decision = "REJECT_TOC_USE_FALLBACK"
                review_reason = "accept_ratio_below_threshold"
            elif critical_toc_review and low_rows > 0:
                final_decision = "REJECT_TOC_USE_FALLBACK"
                review_reason = ",".join(toc_reasons[:4]) or "toc_review"

    if final_decision == "REJECT_TOC_USE_FALLBACK":
        index_source = "classification-fallback"
        classified = build_classification_fallback_index(all_pages)
    queue_bucket = "reindex_review" if final_decision in {"REVIEW", "REJECT_TOC_USE_FALLBACK"} else "index"

    save_index(req.pdf_id, classified)
    update_pdf_record(
        req.pdf_id,
        status="index_ready",
        index_ready=True,
        index_source=index_source,
        queue_bucket=queue_bucket,
        review_reason=review_reason,
    )

    try:
        export_path = export_index_json(
            req.pdf_id, record.get("filename", ""), classified
        )
    except Exception as exc:
        log.warning("Index JSON export failed (non-fatal): %s", exc)
        export_path = ""

    record = get_pdf_record(req.pdf_id)
    stage_timings["total_generate_index_s"] = round(time.perf_counter() - req_t0, 3)
    stage_timings["toc_vision_calls"] = int(toc_stage_stats.get("toc_vision_calls", 0))
    stage_timings["toc_vision_time_s"] = round(float(toc_stage_stats.get("toc_vision_time_s", 0.0)), 3)
    stage_timings["toc_text_calls"] = int(toc_stage_stats.get("toc_text_calls", 0))
    stage_timings["toc_text_time_s"] = round(float(toc_stage_stats.get("toc_text_time_s", 0.0)), 3)
    stage_timings["toc_text_failures"] = int(toc_stage_stats.get("toc_text_failures", 0))
    stage_timings["toc_vision_failures"] = int(toc_stage_stats.get("toc_vision_failures", 0))
    stage_timings["final_decision"] = final_decision
    log.info("Index stage timings: %s", stage_timings)
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
        "index_source":        index_source,
        "export_path":         export_path,
        "status":              record["status"],
        "retrieval_status":    record["retrieval_status"],
        "pending_pages":       record["pending_pages"],
        "chat_ready":          record["chat_ready"],
        "stage_timings":       stage_timings,
        "toc_quality":         {k: v for k, v in toc_quality.items() if k != "items"},
        "row_quality":         {k: v for k, v in row_quality.items() if k != "items"},
        "final_decision":      final_decision,
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


@app.get("/api/golden-eval")
async def golden_eval(limit: int = 50):
    specs = _load_golden_specs(limit=limit)
    if not specs:
        return {
            "count": 0,
            "exact_match_rate": 0.0,
            "avg_f1": 0.0,
            "details": [],
            "message": f"No golden specs found in {GOLDEN_SET_DIR}",
        }

    details: list[dict] = []
    f1_total = 0.0
    exact_hits = 0
    evaluated = 0
    for spec in specs:
        saved = get_saved_index(spec["pdf_id"])
        if not saved:
            details.append({
                "pdf_id": spec["pdf_id"],
                "spec": spec["name"],
                "status": "missing_saved_index",
            })
            continue
        score = _evaluate_index_accuracy(saved, spec["expected_index"])
        evaluated += 1
        f1_total += float(score["f1"])
        if score["exact_match"]:
            exact_hits += 1
        details.append({
            "pdf_id": spec["pdf_id"],
            "spec": spec["name"],
            "status": "evaluated",
            **score,
        })

    return {
        "count": evaluated,
        "exact_match_rate": round(exact_hits / max(1, evaluated), 3),
        "avg_f1": round(f1_total / max(1, evaluated), 3),
        "details": details,
    }


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
    if ENABLE_WARM_STARTUP:
        warm_t0 = time.perf_counter()
        try:
            get_embedder()
            log.info("Warmup: embedding model ready")
        except Exception as exc:
            log.warning("Warmup: embedding preload failed: %s", exc)
        try:
            _ = call_text_llm(
                [{"role": "user", "content": "Respond with: ok"}],
                max_tokens=8,
                temperature=0.0,
                timeout_s=min(8.0, WARM_STARTUP_TIMEOUT_S),
            )
            log.info("Warmup: text model ping done")
        except Exception as exc:
            log.warning("Warmup: text model ping failed: %s", exc)
        if ENABLE_VISION:
            try:
                tiny_img = Image.new("RGB", (32, 32), color=(255, 255, 255))
                _ = call_vision_llm(
                    image_to_jpeg_b64(tiny_img, max_side=64, quality=60),
                    "Say ok.",
                    max_tokens=8,
                    timeout_s=min(10.0, WARM_STARTUP_TIMEOUT_S),
                )
                log.info("Warmup: vision model ping done")
            except Exception as exc:
                log.warning("Warmup: vision model ping failed: %s", exc)
        log.info("Warmup total: %.2fs", time.perf_counter() - warm_t0)
    log.info("Server ready âœ“")
