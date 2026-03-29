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
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from pathlib import Path
from io import BytesIO
from threading import Lock, Thread
from typing import Optional

try:
    import torch
except Exception:
    torch = None

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
    list_reindex_review_pdf_ids,
    list_stage1_batch_pdf_ids,
    replace_extracted_pages,
    save_index,
    update_pdf_record,
    upsert_extracted_pages,
    upsert_pdf_record,
    utc_now_iso,
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
DOCUMENT_CATALOG_UI_PATH = Path(__file__).resolve().parent.parent / "frontend" / "src" / "documentCatalog.js"

LOCAL_LLM_BASE_URL = (os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434") or "http://127.0.0.1:11434").rstrip("/")
LOCAL_TEXT_MODEL = os.getenv("LOCAL_TEXT_MODEL", "qwen2.5:14b")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "qwen2.5vl:7b")
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "180"))
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_STORAGE_PATH = os.getenv("PDF_STORAGE_PATH", "./stored_pdfs")
INDEX_EXPORT_PATH = os.getenv("INDEX_EXPORT_PATH", "./index_exports")
TIMING_LOG_PATH = os.getenv("TIMING_LOG_PATH", "./timing_logs")
BATCH_REPORT_PATH = os.getenv("BATCH_REPORT_PATH", "./batch_reports")
TESSERACT_LANG   = os.getenv("TESSERACT_LANG", "hin+eng")
ENABLE_HANDWRITTEN_HINDI_ASSIST = os.getenv("ENABLE_HANDWRITTEN_HINDI_ASSIST", "true").lower() != "false"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_BATCH_SIZE = max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))
VECTOR_DB_BATCH_SIZE = max(1, int(os.getenv("VECTOR_DB_BATCH_SIZE", "128")))
OCR_WORKER_COUNT = max(1, int(os.getenv("OCR_WORKER_COUNT", "4")))
PREFER_CUDA_EMBEDDINGS = os.getenv("PREFER_CUDA_EMBEDDINGS", "true").lower() != "false"
try:
    PARENT_DOCUMENT_CATALOG = json.loads(DOCUMENT_CATALOG_PATH.read_text(encoding="utf-8"))
except Exception:
    PARENT_DOCUMENT_CATALOG = []
try:
    catalog_js = DOCUMENT_CATALOG_UI_PATH.read_text(encoding="utf-8")
    catalog_js = catalog_js.replace("export const documentCatalog =", "", 1).strip()
    if catalog_js.endswith(";"):
        catalog_js = catalog_js[:-1].strip()
    FULL_DOCUMENT_CATALOG = json.loads(catalog_js)
except Exception:
    FULL_DOCUMENT_CATALOG = [
        {**item, "subDocuments": []}
        for item in PARENT_DOCUMENT_CATALOG
    ]
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
embedder_device = "cpu"
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
    "stop_requested": False,
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
    "stop_requested": False,
}

stage1_batch_runner_lock = Lock()
stage1_batch_runner_status = {
    "running": False,
    "processed": 0,
    "total": 0,
    "current_pdf_id": "",
    "current_filename": "",
    "last_error": "",
    "heartbeat_ts": time.time(),
    "started_at": 0.0,
    "status": "idle",
    "stop_requested": False,
}

audit_runner_lock = Lock()
audit_runner_status = {
    "running": False,
    "processed": 0,
    "total": 0,
    "flagged": 0,
    "current_pdf_id": "",
    "current_filename": "",
    "last_error": "",
    "heartbeat_ts": time.time(),
    "started_at": 0.0,
    "status": "idle",
    "stop_requested": False,
}

reindex_review_runner_lock = Lock()
reindex_review_runner_status = {
    "running": False,
    "processed": 0,
    "total": 0,
    "fixed": 0,
    "current_pdf_id": "",
    "current_filename": "",
    "last_error": "",
    "heartbeat_ts": time.time(),
    "started_at": 0.0,
    "status": "idle",
    "stop_requested": False,
}

pdf_timing_history: dict[str, list[dict]] = {}
pdf_timing_lock = Lock()


def timing_log_path(pdf_id: str) -> Path:
    return Path(TIMING_LOG_PATH) / f"{pdf_id}.json"


def record_pdf_timing_run(pdf_id: str, filename: str, run_label: str, timings: dict[str, float]):
    event = {
        "recorded_at": utc_now_iso(),
        "filename": filename or "",
        "run_label": run_label,
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "total_seconds": round(sum(timings.values()), 3),
    }
    with pdf_timing_lock:
        history = pdf_timing_history.setdefault(pdf_id, [])
        history.append(event)
        if len(history) > 20:
            del history[:-20]
        try:
            timing_log_path(pdf_id).write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            log.exception("Failed to persist timing history for %s", pdf_id)
    update_batch_report_for_pdf(pdf_id)


def get_pdf_timing_history(pdf_id: str) -> list[dict]:
    with pdf_timing_lock:
        history = pdf_timing_history.get(pdf_id)
        if history is not None:
            return list(history)
        log_path = timing_log_path(pdf_id)
        if not log_path.exists():
            return []
        try:
            loaded = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read timing history for %s", pdf_id)
            return []
        if not isinstance(loaded, list):
            return []
        normalized = loaded[-20:]
        pdf_timing_history[pdf_id] = normalized
        return list(normalized)


def batch_report_path(batch_run_id: str) -> Path:
    return Path(BATCH_REPORT_PATH) / f"{sanitize_export_stem(batch_run_id)}.json"


def create_batch_run_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = hashlib.md5(f"{stamp}-{time.time()}".encode("utf-8")).hexdigest()[:6]
    return f"batch_{stamp}_{suffix}"


def is_terminal_pdf_status(record: Optional[dict]) -> bool:
    if not record:
        return False
    status = str(record.get("status") or "").lower()
    retrieval = str(record.get("retrieval_status") or "").lower()
    if status in {"queued_for_stage1", "indexing_running", "extracting_text", "vectorizing", "building_index", "full_ingestion_running"}:
        return False
    if retrieval in {"queued_for_stage1", "full_ingestion_running"}:
        return False
    if int(record.get("pending_pages") or 0) > 0 and retrieval not in {"vectorized", "queued_for_full_ingestion"}:
        return False
    return True


def build_batch_pdf_entry(record: dict) -> dict:
    pdf_id = record.get("pdf_id") or ""
    timing_events = get_pdf_timing_history(pdf_id)
    saved_index = get_saved_index(pdf_id)
    source_counts: dict[str, int] = {}
    for item in saved_index:
        source = str(item.get("source") or "unknown").strip() or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
    latest_event = timing_events[-1] if timing_events else {}
    latest_total = float(latest_event.get("total_seconds") or 0.0)
    total_logged = round(sum(float(event.get("total_seconds") or 0.0) for event in timing_events), 3)
    return {
        "pdf_id": pdf_id,
        "filename": record.get("filename") or f"{pdf_id}.pdf",
        "cnr_number": record.get("cnr_number") or "",
        "status": record.get("status") or "",
        "retrieval_status": record.get("retrieval_status") or "",
        "queue_bucket": record.get("queue_bucket") or "",
        "review_reason": record.get("review_reason") or "",
        "index_source": record.get("index_source") or "",
        "total_pages": int(record.get("total_pages") or 0),
        "indexed_pages": int(record.get("indexed_pages") or 0),
        "pending_pages": int(record.get("pending_pages") or 0),
        "batch_enqueued_at": record.get("batch_enqueued_at") or "",
        "updated_at": record.get("updated_at") or "",
        "is_terminal": is_terminal_pdf_status(record),
        "index_entries": len(saved_index),
        "index_source_breakdown": source_counts,
        "latest_run_label": latest_event.get("run_label") or "",
        "latest_run_total_seconds": round(latest_total, 3),
        "timing_event_count": len(timing_events),
        "logged_total_seconds": total_logged,
        "timings": timing_events,
    }


def write_batch_report(batch_run_id: str) -> Optional[Path]:
    batch_run_id = str(batch_run_id or "").strip()
    if not batch_run_id:
        return None
    records = [record for record in list_pdf_records() if str(record.get("batch_run_id") or "") == batch_run_id]
    if not records:
        return None
    records.sort(key=lambda item: (str(item.get("batch_enqueued_at") or ""), str(item.get("updated_at") or ""), str(item.get("filename") or "")))
    pdf_entries = [build_batch_pdf_entry(record) for record in records]
    started_candidates = [entry.get("batch_enqueued_at") for entry in pdf_entries if entry.get("batch_enqueued_at")]
    completed_candidates = [entry.get("updated_at") for entry in pdf_entries if entry.get("updated_at") and entry.get("is_terminal")]
    summary = {
        "total_pdfs": len(pdf_entries),
        "completed_pdfs": sum(1 for entry in pdf_entries if entry.get("is_terminal")),
        "review_pdfs": sum(1 for entry in pdf_entries if entry.get("review_reason")),
        "failed_pdfs": sum(1 for entry in pdf_entries if str(entry.get("status") or "").lower() == "failed"),
        "vectorized_pdfs": sum(1 for entry in pdf_entries if str(entry.get("retrieval_status") or "").lower() == "vectorized"),
        "finalized_pdfs": sum(1 for entry in pdf_entries if str(entry.get("status") or "").lower() in {"vectorized", "index_ready", "needs_review"}),
        "total_logged_seconds": round(sum(float(entry.get("logged_total_seconds") or 0.0) for entry in pdf_entries), 3),
    }
    payload = {
        "batch_run_id": batch_run_id,
        "generated_at": utc_now_iso(),
        "started_at": min(started_candidates) if started_candidates else "",
        "last_completed_at": max(completed_candidates) if completed_candidates else "",
        "status": "completed" if summary["completed_pdfs"] == summary["total_pdfs"] else "running",
        "summary": summary,
        "pdfs": pdf_entries,
    }
    report_path = batch_report_path(batch_run_id)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def update_batch_report_for_pdf(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        return
    batch_run_id = str(record.get("batch_run_id") or "").strip()
    if batch_run_id:
        try:
            write_batch_report(batch_run_id)
        except Exception:
            log.exception("Failed to update batch report for %s", pdf_id)


def list_batch_reports(limit: int = 12) -> list[dict]:
    reports = []
    report_dir = Path(BATCH_REPORT_PATH)
    if not report_dir.exists():
        return reports
    for report_file in sorted(report_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        try:
            payload = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read batch report %s", report_file)
            continue
        reports.append({
            "batch_run_id": payload.get("batch_run_id") or report_file.stem,
            "generated_at": payload.get("generated_at") or "",
            "started_at": payload.get("started_at") or "",
            "last_completed_at": payload.get("last_completed_at") or "",
            "status": payload.get("status") or "unknown",
            "summary": payload.get("summary") or {},
            "path": str(report_file),
        })
    return reports

# ── ChromaDB ──────────────────────────────────────────────────────────────────
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
Path(PDF_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
Path(INDEX_EXPORT_PATH).mkdir(parents=True, exist_ok=True)
Path(TIMING_LOG_PATH).mkdir(parents=True, exist_ok=True)
Path(BATCH_REPORT_PATH).mkdir(parents=True, exist_ok=True)
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
        record_pdf_timing_run(self.pdf_id, self.filename, run_label, self.timings)

def pdf_id_from_bytes(data: bytes) -> str:
    """Stable ID for a PDF based on its content hash."""
    return hashlib.md5(data).hexdigest()[:16]


def stored_pdf_path(pdf_id: str) -> Path:
    """Local on-disk copy of the uploaded PDF for later page-image reprocessing."""
    return Path(PDF_STORAGE_PATH) / f"{pdf_id}.pdf"


def sanitize_export_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\|?*\x00-\x1f]+', "_", (value or "").strip())
    return cleaned.strip(" .") or "index"


def normalize_catalog_label(value: str = "") -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


CATALOG_PARENT_LOOKUP = {
    normalize_catalog_label(item.get("name", "")): item
    for item in FULL_DOCUMENT_CATALOG
    if item.get("name")
}
CATALOG_SUBDOC_LOOKUP = {
    normalize_catalog_label(item.get("name", "")): {
        normalize_catalog_label(sub_item.get("name", "")): sub_item
        for sub_item in (item.get("subDocuments") or [])
        if sub_item.get("name")
    }
    for item in FULL_DOCUMENT_CATALOG
    if item.get("name")
}
CATALOG_OTHERS_PARENT = CATALOG_PARENT_LOOKUP.get("others") or CATALOG_PARENT_LOOKUP.get("other") or {
    "code": "13",
    "name": "Others",
    "subDocuments": [],
}


def derive_case_metadata(value: str = "") -> dict:
    stem = Path(value or "").stem.strip()
    match = re.search(r"\b([A-Za-z]+)[_-](\d+)[_-](\d{4})\b", stem)
    if not match:
        normalized = stem.replace("-", "_").replace(" ", "_").upper()
        return {
            "case_no": normalized or stem or "",
            "case_type": "",
            "case_number": "",
            "case_year": "",
        }

    case_type = match.group(1).upper()
    case_number = match.group(2)
    case_year = match.group(3)
    return {
        "case_no": f"{case_type}_{case_number}_{case_year}",
        "case_type": case_type,
        "case_number": case_number,
        "case_year": case_year,
    }


def resolve_export_document_fields(item: dict) -> dict:
    raw_title = str(item.get("title") or item.get("displayTitle") or item.get("originalTitle") or "").strip()
    raw_subdocument = str(item.get("subDocument") or "").strip()
    parent = CATALOG_PARENT_LOOKUP.get(normalize_catalog_label(raw_title)) or CATALOG_OTHERS_PARENT
    doc_code = str(parent.get("code") or CATALOG_OTHERS_PARENT.get("code") or "")
    doc_name = str(parent.get("name") or raw_title or "Others").strip() or "Others"

    sub_lookup = CATALOG_SUBDOC_LOOKUP.get(normalize_catalog_label(doc_name), {})
    sub_item = None
    if raw_subdocument:
        sub_item = sub_lookup.get(normalize_catalog_label(raw_subdocument))
    if not sub_item and sub_lookup:
        sub_item = sub_lookup.get("others") or sub_lookup.get("other")

    if sub_item:
        doc_subcode = str(sub_item.get("code") or "")
        doc_subname = str(sub_item.get("name") or raw_subdocument or "Others").strip() or "Others"
    else:
        doc_subcode = ""
        doc_subname = raw_subdocument or "Others"

    if parent is CATALOG_OTHERS_PARENT and not raw_title:
        raw_title = "Others"

    return {
        "doc_code": doc_code,
        "doc_name": doc_name,
        "doc_subcode": doc_subcode,
        "doc_subname": doc_subname,
        "raw_title": raw_title or doc_name,
        "raw_subdocument": raw_subdocument,
    }


def build_export_document_rows(index_items: list[dict], case_meta: dict) -> list[dict]:
    rows = []
    for item in (index_items or []):
        doc_fields = resolve_export_document_fields(item)
        rows.append({
            "case_no": case_meta.get("case_no", ""),
            "case_type": case_meta.get("case_type", ""),
            "case_number": case_meta.get("case_number", ""),
            "case_year": case_meta.get("case_year", ""),
            **doc_fields,
            "page_from": item.get("pdfPageFrom", item.get("pageFrom")),
            "page_to": item.get("pdfPageTo", item.get("pageTo")),
            "toc_page_from": item.get("tocPageFrom", item.get("pageFrom")),
            "toc_page_to": item.get("tocPageTo", item.get("pageTo")),
            "receiving_date": str(item.get("receivingDate") or "").strip(),
            "serial_no": str(item.get("serialNo") or "").strip(),
            "court_fee": str(item.get("courtFee") or "").strip(),
            "source": str(item.get("source") or "").strip(),
            "note": str(item.get("note") or "").strip(),
            "batch_no": str(item.get("batchNo") or "").strip(),
        })
    return rows


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
    case_meta = derive_case_metadata((record or {}).get("filename") or cnr_number)
    export_rows = build_export_document_rows(index_items, case_meta)
    payload = {
        "case_no": case_meta.get("case_no", ""),
        "case_type": case_meta.get("case_type", ""),
        "case_number": case_meta.get("case_number", ""),
        "case_year": case_meta.get("case_year", ""),
        "pdf_id": pdf_id,
        "cnr_number": cnr_number,
        "filename": (record or {}).get("filename", ""),
        "total_pages": total_pages,
        "indexed_page_start": indexed_start,
        "indexed_page_end": indexed_end,
        "indexed_pages": max(indexed_end - indexed_start + 1, 0),
        "index_source": index_source,
        "status": (record or {}).get("status", "index_ready"),
        "index_entries": len(export_rows),
        "exported_at": utc_now_iso(),
        "documents": export_rows,
    }
    export_stem = case_meta.get("case_no") or cnr_number or pdf_id
    export_path = Path(INDEX_EXPORT_PATH) / f"{sanitize_export_stem(export_stem)}.json"
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


def resolve_embedding_device() -> str:
    if PREFER_CUDA_EMBEDDINGS and torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def get_embedder_device() -> str:
    return embedder_device or "cpu"


def get_embedder():
    """Load the embedding model once and keep it pinned to the preferred device."""
    global embedder, embedder_device
    if embedder is None:
        with embedder_lock:
            if embedder is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    preferred_device = resolve_embedding_device()
                    log.info("Loading embedding model '%s' on device=%s", EMBEDDING_MODEL_NAME, preferred_device)
                    embedder = SentenceTransformer(
                        EMBEDDING_MODEL_NAME,
                        cache_folder=HF_CACHE_PATH,
                        device=preferred_device,
                    )
                    embedder_device = str(getattr(embedder, "device", None) or getattr(embedder, "_target_device", preferred_device) or preferred_device)
                    log.info("Embedding model ready on device=%s", embedder_device)
                except Exception as e:
                    embedder = False
                    embedder_device = "fallback-cpu"
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


def embed_texts(texts: list[str], batch_size: Optional[int] = None, pdf_id: str = "") -> list[list[float]]:
    """Embed a list of strings using the local multilingual model in GPU-friendly batches."""
    model = get_embedder()
    effective_batch_size = max(1, int(batch_size or EMBEDDING_BATCH_SIZE))
    if not texts:
        return []

    log.info(
        "[VECTORIZE] pdf=%s total_chunks=%s embedding_device=%s embedding_batch_size=%s",
        pdf_id or "-",
        len(texts),
        get_embedder_device(),
        effective_batch_size,
    )

    embeddings: list[list[float]] = []
    total_batches = math.ceil(len(texts) / effective_batch_size)
    for batch_index in range(0, len(texts), effective_batch_size):
        batch_texts = texts[batch_index:batch_index + effective_batch_size]
        started = time.perf_counter()
        if model is False:
            batch_vectors = fallback_embed_texts(batch_texts)
        else:
            batch_vectors = model.encode(
                batch_texts,
                batch_size=effective_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()
        elapsed = time.perf_counter() - started
        log.info(
            "[VECTORIZE] pdf=%s embedding batch %s/%s size=%s took %.3fs",
            pdf_id or "-",
            (batch_index // effective_batch_size) + 1,
            total_batches,
            len(batch_texts),
            elapsed,
        )
        embeddings.extend(batch_vectors)
    return embeddings


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


DEVANAGARI_DIGIT_MAP = str.maketrans("\u0966\u0967\u0968\u0969\u096A\u096B\u096C\u096D\u096E\u096F", "0123456789")


def debug_dump(label: str, value, max_chars: int = 6000):
    text_value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    text_value = text_value or ""
    clipped = text_value[:max_chars]
    if len(text_value) > max_chars:
        clipped += f"\n...[truncated {len(text_value) - max_chars} chars]"
    log.info("[DEBUG DUMP] %s:\n%s", label, clipped)


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def normalize_page_digits(text: str) -> str:
    return (text or "").translate(DEVANAGARI_DIGIT_MAP)


def needs_ocr(direct_text: str) -> bool:
    stats = analyze_extracted_text(direct_text)
    if stats["chars"] == 0:
        return True
    if stats["words"] >= 18 and stats["line_count"] >= 3:
        return False
    if stats["chars"] >= 120 and stats["line_count"] >= 2:
        return False
    return True


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
    Falls back to OCR only when the direct text looks incomplete.
    For hard handwritten/low-quality Hindi pages, optionally uses the vision model
    to improve the stored text used by indexing and chat.
    """
    direct_text = normalize_ocr_text(page.get_text("text"))
    debug_dump(f"direct text page {page_num}", direct_text)

    if not needs_ocr(direct_text):
        log.info("[OCR GATE] Page %s skipped OCR and used direct text (%s chars)", page_num, len(direct_text))
        return {
            "text": direct_text,
            "used_ocr": False,
            "vision_used": False,
            "handwriting_suspected": False,
            "extraction_method": "digital",
        }

    log.info("[OCR GATE] Page %s sent to OCR (%s chars of direct text)", page_num, len(direct_text))
    ocr_started = time.perf_counter()
    image = render_page_image(page, dpi=dpi)
    ocr_text = normalize_ocr_text(ocr_page_image(image))
    debug_dump(f"ocr text page {page_num}", ocr_text)
    vision_used = False
    handwriting_suspected = should_try_handwritten_assist(direct_text, ocr_text)
    final_text = ocr_text
    extraction_method = "ocr"

    if handwriting_suspected:
        enhanced_text = extract_handwritten_page_text(image, page_num)
        if enhanced_text:
            enhanced_text = normalize_ocr_text(enhanced_text)
            enhanced_stats = analyze_extracted_text(enhanced_text)
            ocr_stats = analyze_extracted_text(ocr_text)
            if enhanced_stats["chars"] >= max(ocr_stats["chars"], 80):
                final_text = enhanced_text
                vision_used = True
                extraction_method = "vision_ocr"

    if timing_collector:
        timing_collector.add_duration("ocr_time", time.perf_counter() - ocr_started)

    return {
        "text": final_text or direct_text,
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
- Do not invent one broad summary row for the whole PDF; if rows are unreadable, return [].
- Prefer exact row splitting, wrapped-line merging, and page-number column reading over guessing.

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


def parse_raw_index_items(items: list[dict], default_source: str) -> list[dict]:
    parsed = []
    for item in items:
        pf = coerce_page_number(item.get("pageFrom"), 1)
        pt = coerce_page_number(item.get("pageTo"), pf)
        if pt < pf:
            pt = pf
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        parsed.append({
            "title": title,
            "displayTitle": str(item.get("displayTitle") or item.get("originalTitle") or title).strip(),
            "originalTitle": str(item.get("originalTitle") or title).strip(),
            "pageFrom": pf,
            "pageTo": pt,
            "tocPageFrom": pf,
            "tocPageTo": pt,
            "source": item.get("source", default_source),
            "serialNo": str(item.get("serialNo", "")),
            "courtFee": str(item.get("courtFee", "")),
        })
    return parsed


def is_toc_anchor_title(raw_title: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", (raw_title or "").lower())
    return normalized in {"index", "tableofcontents", "contents", "listofdocuments"}


def infer_toc_page_offset(raw_items: list[dict], toc_page_nums: list[int], indexed_start: int, range_end: int) -> tuple[int, str]:
    if not raw_items:
        return 0, "no-items"

    primary_toc_page = min([page_num for page_num in toc_page_nums if isinstance(page_num, int)], default=indexed_start)
    structural_offsets = []
    for item in raw_items:
        page_from = item["pageFrom"]
        if page_from < 1:
            continue
        if is_toc_anchor_title(item["title"]):
            structural_offsets.append(primary_toc_page - page_from)

    min_raw_page = min(item["pageFrom"] for item in raw_items)
    candidate_offsets = [0]
    candidate_offsets.extend(offset for offset in structural_offsets if offset > 0)
    if indexed_start > 1 and min_raw_page >= 1:
        heuristic_offset = indexed_start - min_raw_page
        if heuristic_offset > 0:
            candidate_offsets.append(heuristic_offset)

    best_offset = 0
    best_reason = "absolute"
    best_score = float("-inf")

    for offset in dict.fromkeys(candidate_offsets):
        mapped_pages = []
        valid = True
        for item in raw_items:
            mapped_from = item["pageFrom"] + offset
            mapped_to = item["pageTo"] + offset
            if mapped_from < 1 or mapped_to < mapped_from or mapped_to > range_end:
                valid = False
                break
            mapped_pages.append((mapped_from, mapped_to))
        if not valid:
            continue

        score = 0
        reason = "absolute"
        if offset == 0:
            score += 1
        if offset in structural_offsets:
            score += 8
            reason = "structural-offset"
        if indexed_start > 1 and min_raw_page < indexed_start and offset == indexed_start - min_raw_page:
            score += 3
            reason = "heuristic-offset" if reason == "absolute" else reason
        if offset > 0 and min_raw_page <= 3:
            score += 2
        if offset > 0 and mapped_pages and min(page_from for page_from, _ in mapped_pages) >= indexed_start:
            score += 1

        if score > best_score:
            best_score = score
            best_offset = offset
            best_reason = reason

    if best_offset == 0:
        return 0, best_reason
    if best_reason == "absolute":
        return 0, "absolute"
    return best_offset, best_reason


def normalize_index_items(
    items: list[dict],
    indexed_start: int,
    indexed_end: int,
    default_source: str,
    toc_page_nums: Optional[list[int]] = None,
) -> list[dict]:
    raw_items = parse_raw_index_items(items, default_source)
    inferred_offset, offset_reason = infer_toc_page_offset(raw_items, toc_page_nums or [], indexed_start, indexed_end)
    lower_bound = 1 if inferred_offset > 0 else indexed_start
    if inferred_offset > 0:
        log.info(
            "Applying TOC page offset=%s for TOC pages=%s indexed_start=%s reason=%s",
            inferred_offset,
            toc_page_nums or [],
            indexed_start,
            offset_reason,
        )

    normalized = []
    for item in raw_items:
        pf = item["pageFrom"] + inferred_offset
        pt = item["pageTo"] + inferred_offset
        pf = max(lower_bound, min(pf, indexed_end))
        pt = max(pf, min(pt, indexed_end))
        normalized.append({
            **item,
            "pageFrom": pf,
            "pageTo": pt,
            "pdfPageFrom": pf,
            "pdfPageTo": pt,
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


def build_toc_ranges_from_items(
    items: list[dict],
    indexed_start: int,
    range_end: int,
    default_source: str,
    toc_page_nums: Optional[list[int]] = None,
) -> list[dict]:
    normalized = normalize_index_items(items, indexed_start, range_end, default_source, toc_page_nums=toc_page_nums)
    if not normalized:
        return []

    ranged = []
    for idx, item in enumerate(normalized):
        current = dict(item)
        next_start = normalized[idx + 1]["pageFrom"] if idx + 1 < len(normalized) else None
        next_toc_start = normalized[idx + 1].get("tocPageFrom") if idx + 1 < len(normalized) else None
        if next_start is not None and next_start > current["pageFrom"]:
            current["pageTo"] = max(current["pageFrom"], next_start - 1)
            current["pdfPageTo"] = current["pageTo"]
        else:
            current["pageTo"] = max(current["pageFrom"], min(current["pageTo"], range_end))
            current["pdfPageTo"] = current["pageTo"]
        current["pageTo"] = min(current["pageTo"], range_end)
        current["pdfPageTo"] = min(current.get("pdfPageTo", current["pageTo"]), range_end)
        if next_toc_start is not None and next_toc_start > current.get("tocPageFrom", current["pageFrom"]):
            current["tocPageTo"] = max(current.get("tocPageFrom", current["pageFrom"]), next_toc_start - 1)
        else:
            current["tocPageTo"] = max(current.get("tocPageFrom", current["pageFrom"]), current.get("tocPageTo", current.get("tocPageFrom", current["pageFrom"])))
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


def score_toc_features(features: dict) -> int:
    return (
        (features["marker_hits"] * 12)
        + (features["header_hits"] * 10)
        + (features["line_hits"] * 6)
        + (features["numbered_row_lines"] * 5)
        + (features["page_range_hits"] * 4)
        + (features["short_row_like_lines"] * 3)
        + (features["row_like_lines"] * 2)
        + features["dotted_leaders"]
    )


TOC_POSITIVE_PAGE_PATTERNS = [
    (r"\bindex\b", 90),
    (r"table of contents", 90),
    (r"\bs\.?\s*no\b", 35),
    (r"\bdescription of (?:the )?documents\b", 55),
    (r"\bannexures?\b", 35),
    (r"\bpages?\b", 20),
    (r"\bchronology of events\b", 12),
]
TOC_NEGATIVE_PAGE_PATTERNS = [
    (r"\bscrutiny report\b", 120),
    (r"\bcomputer sheet\b", 90),
    (r"\boffice note\b", 75),
    (r"\bthe defaults?\b", 60),
    (r"\bcourt fees?\b", 55),
    (r"\bdescription of relief\b", 45),
    (r"\blistable before\b", 35),
]


def explicit_toc_page_score(text: str, features: dict) -> int:
    lower = (text or "").lower()
    score = score_toc_features(features)
    for pattern, weight in TOC_POSITIVE_PAGE_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            score += weight
    for pattern, penalty in TOC_NEGATIVE_PAGE_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            score -= penalty
    if features.get("header_hits", 0) >= 2 and features.get("numbered_row_lines", 0) >= 2:
        score += 40
    if features.get("marker_hits", 0) >= 1 and features.get("header_hits", 0) >= 1:
        score += 35
    return score


def has_explicit_toc_signal(text: str, features: dict) -> bool:
    lower = (text or "").lower()
    return (
        bool(re.search(r"\bindex\b", lower, flags=re.IGNORECASE))
        or bool(re.search(r"table of contents", lower, flags=re.IGNORECASE))
        or (
            bool(re.search(r"\bdescription of (?:the )?documents\b", lower, flags=re.IGNORECASE))
            and bool(re.search(r"\bannexures?\b", lower, flags=re.IGNORECASE))
        )
        or (features.get("header_hits", 0) >= 2 and features.get("numbered_row_lines", 0) >= 2)
    )


def toc_acceptance_floor(quality: dict, candidate_pages: list[dict]) -> bool:
    kept = int(quality.get("kept_items") or 0)
    ascending = float(quality.get("ascending_ratio") or 0.0)
    coverage = float(quality.get("coverage_ratio") or 0.0)
    explicit_pages = sum(1 for page in candidate_pages if page.get("explicit"))
    strong_explicit = any(
        page.get("explicit") and page.get("features", {}).get("header_hits", 0) >= 1 and page.get("features", {}).get("numbered_row_lines", 0) >= 2
        for page in candidate_pages
    )

    if explicit_pages >= 1:
        if kept >= 3 and ascending >= 0.90 and coverage >= 0.75:
            return True
        if kept >= 2 and strong_explicit and ascending >= 1.0 and coverage >= 0.55:
            return True
        return False

    if kept >= 3 and ascending >= 0.66 and coverage >= 0.5:
        return True
    return False


def should_force_image_toc_retry(candidate_pages: list[dict], result: dict, toc_markers: list[str], rule_quality: Optional[dict] = None) -> bool:
    if not candidate_pages:
        return False

    ocr_pages = len(result.get("ocr_pages") or [])
    candidate_line_count = int(result.get("candidate_line_count") or 0)
    explicit_signal = False
    for page in candidate_pages:
        page_text = page.get("text", "")
        features = analyze_toc_page_features(page_text, toc_markers)
        if has_explicit_toc_signal(page_text, features):
            explicit_signal = True
            break

    scanned_or_noisy = ocr_pages >= max(1, len(candidate_pages)) or any(page.get("vision_used") for page in candidate_pages)
    strong_rule = bool(
        rule_quality
        and rule_quality.get("accepted")
        and int(rule_quality.get("kept_items") or 0) >= 3
        and float(rule_quality.get("coverage_ratio") or 0.0) >= 0.75
    )
    weak_rule = (
        not rule_quality
        or not rule_quality.get("accepted")
        or int(rule_quality.get("kept_items") or 0) < 3
        or float(rule_quality.get("coverage_ratio") or 0.0) < 0.80
    )
    sparse_lines = candidate_line_count <= 14
    return explicit_signal and (scanned_or_noisy or sparse_lines) and not strong_rule and weak_rule


def build_strict_index_review_reasons(total_pages: int, explicit_toc_detected: bool, toc_items: list[dict], final_items: list[dict]) -> list[str]:
    non_gap_items = [item for item in final_items if str(item.get("source") or "").strip() != "gap"]
    auto_items = [item for item in non_gap_items if str(item.get("source") or "").strip() == "auto"]
    reasons = []

    if explicit_toc_detected and not toc_items:
        reasons.append("explicit_toc_detected_but_not_extracted")
    if explicit_toc_detected and auto_items:
        reasons.append("toc_detected_but_final_index_used_ai_fallback")
    if total_pages >= 6 and len(non_gap_items) <= 1 and auto_items:
        reasons.append("single_ai_section_for_multi_page_pdf")
    if explicit_toc_detected and total_pages >= 6 and len(non_gap_items) <= 1:
        reasons.append("too_few_sections_for_visible_toc")

    return list(dict.fromkeys(reasons))


def rank_toc_candidate_pages(candidate_pages: list[dict], toc_markers: list[str]) -> list[dict]:
    ranked = []
    for page in candidate_pages:
        page_text = page.get("text", "")
        features = analyze_toc_page_features(page_text, toc_markers)
        ranked.append({
            "page": page,
            "score": explicit_toc_page_score(page_text, features),
            "features": features,
            "explicit": has_explicit_toc_signal(page_text, features),
        })
    ranked.sort(key=lambda item: (-item["score"], not item["explicit"], item["page"]["page_num"]))
    return ranked


def select_toc_pages_for_extraction(ranked_pages: list[dict], max_pages: int = 3) -> list[dict]:
    if not ranked_pages:
        return []

    ranked_by_priority = sorted(
        ranked_pages,
        key=lambda item: (-int(bool(item.get("explicit"))), -item["score"], item["page"]["page_num"])
    )
    best_anchor = ranked_by_priority[0]
    anchor_page_num = best_anchor["page"]["page_num"]

    contiguous = [best_anchor]
    next_page_num = anchor_page_num + 1
    while len(contiguous) < max_pages:
        match = next((item for item in ranked_pages if item["page"]["page_num"] == next_page_num), None)
        if not match:
            break
        contiguous.append(match)
        next_page_num += 1

    return [item["page"] for item in contiguous[:max_pages]]


def is_toc_header_line(line: str, toc_markers: list[str]) -> bool:
    lower = (line or "").strip().lower()
    if re.match(r"^\d{1,3}[\)\.\-: ]+", lower):
        return False
    if re.search(r"\d{1,4}(?:\s*[\-–—]\s*\d{1,4})?\s*$", lower) and len(lower.split()) > 1:
        return False
    return lower in toc_markers or any(
        re.search(pattern, lower, flags=re.IGNORECASE)
        for pattern in TOC_TABLE_HEADER_PATTERNS
    )


TOC_SECTION_STOP_MARKERS = {
    "chronology of events",
    "list of dates and events",
    "dates and events",
}
TOC_INDEX_SECTION_MARKERS = {
    "part a",
    "index",
    "part a - index",
    "part-a",
}
TOC_INDEX_SECTION_STOP_MARKERS = {
    "part b",
    "chronology of events",
    "part b - chronology of events",
    "part-b",
}
TOC_NOISE_MARKERS = {
    "petitioner",
    "respondent",
    "versus",
    "appellant",
    "applicant",
    "non-applicant",
    "high court of",
    "principal seat",
    "miscellaneous petition no",
    "miscellaneous petition",
    "case no",
    "advocate",
    "through",
    "received",
    "advance copy",
    "office of",
    "clerk",
    "counsel for",
    "email",
    "mob.",
    "mobile",
}


def is_toc_noise_line(line: str) -> bool:
    lower = (line or "").lower()
    return any(marker in lower for marker in TOC_NOISE_MARKERS)


def is_toc_stop_line(line: str) -> bool:
    lower = (line or "").strip().lower()
    if not lower or re.search(r"\d", lower):
        return False
    return lower in TOC_SECTION_STOP_MARKERS


def isolate_toc_text_block(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    if not lines:
        return ""

    start_idx = 0
    for idx, line in enumerate(lines):
        lower = line.strip().lower()
        if any(marker in lower for marker in TOC_INDEX_SECTION_MARKERS):
            start_idx = idx
            break

    selected = []
    seen_index_signal = False
    for line in lines[start_idx:]:
        lower = line.strip().lower()
        if any(marker in lower for marker in TOC_INDEX_SECTION_MARKERS):
            seen_index_signal = True
        if seen_index_signal and any(marker in lower for marker in TOC_INDEX_SECTION_STOP_MARKERS):
            break
        selected.append(line)

    block = "\n".join(selected).strip()
    return block or normalize_ocr_text(text)


def clean_toc_title(title: str) -> str:
    cleaned = normalize_page_digits(re.sub(r"\s+", " ", title or "")).strip(" .:-|\t")
    cleaned = re.sub(r"\b(?:annexure|sheet\s*count|court\s*fee)\b.*$", "", cleaned, flags=re.IGNORECASE).strip(" .:-|\t")
    return cleaned


def normalize_toc_parse_line(raw_line: str) -> str:
    line = normalize_page_digits(re.sub(r"\s+", " ", raw_line or "")).strip()
    line = line.replace("|", " ")
    line = re.sub(r"[??]+", " ", line)
    line = re.sub(r"[??]+", " ", line)
    line = re.sub(r"\s*[._]{3,}\s*", " ", line)
    line = re.sub(r"\s+", " ", line).strip(" .:-|\t")
    return line


def split_toc_page_suffix(body: str) -> tuple[str, Optional[int], Optional[int], str]:
    cleaned = normalize_toc_parse_line(body)
    patterns = [
        r"^(?P<title>.*?)(?:\s+(?P<annex>[A-Za-z]{1,4}[-/]\d{1,4}[A-Za-z]?))?\s+(?P<from>\d{1,4})(?:\s*[\-\u2013\u2014]\s*(?P<to>\d{1,4}))?$",
        r"^(?P<title>.*?)(?:\s+(?P<annex>[A-Za-z]{1,4}[-/]\d{1,4}[A-Za-z]?))?\s+(?P<from>\d{1,4})$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if not match:
            continue
        title = re.sub(r"\.{2,}$", "", (match.group("title") or "")).strip(" .:-|\t")
        page_from = int(match.group("from"))
        page_to = int(match.group("to") or page_from)
        if page_to < page_from:
            page_to = page_from
        return title, page_from, page_to, str(match.group("annex") or "")
    return cleaned, None, None, ""


def is_probable_stamp_noise(line: str) -> bool:
    lower = (line or "").strip().lower()
    if not lower:
        return True
    if lower in {"received", "clerk"}:
        return True
    if re.search(r"(?:office of|advance copy|email|mobile|mob\.?|counsel for)", lower, flags=re.IGNORECASE):
        return True
    if re.match(r"^(?:place|date)\s*[:.-]", lower):
        return True
    return False


def should_append_toc_continuation(line: str) -> bool:
    normalized = normalize_toc_parse_line(line)
    if len(normalized) < 2:
        return False
    if is_toc_header_line(normalized, ["index", "contents", "table of contents"]) or is_probable_stamp_noise(normalized):
        return False
    if re.match(r"^\d{1,3}[\)\.\-: ]+", normalized):
        return False
    if re.fullmatch(r"(?:[A-Za-z]{1,4}[-/]\d{1,4}[A-Za-z]?\s+)?\d{1,4}(?:\s*[\-\u2013\u2014]\s*\d{1,4})?", normalized):
        return False
    if re.search(r"\b(?:page\s*no|pages?|annexure|sheet|court fee)\b", normalized, flags=re.IGNORECASE):
        return False
    return True


def evaluate_toc_items_confidence(items: list[dict], max_page_hint: int) -> dict:
    cleaned_items = []
    rejected_titles = []
    previous_page = 0
    ascending_hits = 0

    for item in items:
        title = clean_toc_title(item.get("title", ""))
        lower = title.lower()
        page_from = coerce_page_number(item.get("pageFrom"), 0)
        page_to = coerce_page_number(item.get("pageTo"), page_from)
        is_noise = (
            len(title) < 4
            or len(title) > 180
            or is_toc_noise_line(title)
            or lower in TOC_SECTION_STOP_MARKERS
            or title.count(":") >= 2
            or bool(re.search(r"\b(?:s\.?\s*no|serial\s*no|page\s*no|pages?|particulars?|annexure|sheet)\b", lower, flags=re.IGNORECASE))
            or page_from < 1
            or page_to < page_from
            or page_from > max_page_hint
            or page_to > max_page_hint
        )
        if is_noise:
            rejected_titles.append(title)
            continue

        cleaned = {
            **item,
            "title": title,
            "displayTitle": clean_toc_title(item.get("displayTitle") or title),
            "originalTitle": clean_toc_title(item.get("originalTitle") or title),
            "pageFrom": page_from,
            "pageTo": page_to,
        }
        if cleaned["pageFrom"] >= previous_page:
            ascending_hits += 1
            previous_page = cleaned["pageFrom"]
        cleaned_items.append(cleaned)

    total_items = len(items)
    kept_items = len(cleaned_items)
    ascending_ratio = ascending_hits / max(kept_items, 1)
    coverage_ratio = kept_items / max(total_items, 1)
    has_meaningful_page_span = any(
        item["pageTo"] > item["pageFrom"] or item["pageFrom"] >= 2
        for item in cleaned_items
    )
    accepted = (
        kept_items >= 3
        and coverage_ratio >= 0.6
        and ascending_ratio >= 0.66
    ) or (
        kept_items == 2
        and total_items == 2
        and ascending_ratio >= 1.0
    ) or (
        kept_items >= 4
        and ascending_ratio >= 1.0
        and has_meaningful_page_span
    )
    return {
        "accepted": accepted,
        "items": cleaned_items,
        "kept_items": kept_items,
        "total_items": total_items,
        "coverage_ratio": coverage_ratio,
        "ascending_ratio": ascending_ratio,
        "rejected_titles": rejected_titles[:8],
    }


def filter_toc_lines(text: str, toc_markers: list[str]) -> list[str]:
    filtered = []
    previous_was_row = False
    seen_table_header = False
    range_sep = r"[\-–—]"
    for raw_line in (text or "").splitlines():
        line = normalize_toc_parse_line(raw_line)
        if len(line) < 1:
            previous_was_row = False
            continue
        if is_toc_stop_line(line):
            break
        if is_probable_stamp_noise(line):
            previous_was_row = False
            continue
        if is_toc_noise_line(line) and not re.search(r"\d", line):
            previous_was_row = False
            continue

        keep = False
        title_part, page_from, _page_to, annex = split_toc_page_suffix(line)
        if is_toc_header_line(line, toc_markers):
            keep = True
            seen_table_header = True
        elif re.match(rf"^\d{{1,3}}[\)\.\-: ]+.+?(?:\.{{2,}}\s*)?\d{{1,4}}(?:\s*{range_sep}\s*\d{{1,4}})?\s*$", line):
            keep = True
        elif seen_table_header and re.match(r"^\d{1,3}[\)\.\-: ]+.+$", line):
            keep = True
        elif page_from is not None and len(title_part) >= 3:
            keep = True
        elif seen_table_header and page_from is not None and annex:
            keep = True
        elif re.fullmatch(rf"(?:[A-Za-z]{{1,4}}[-/]\d{{1,4}}[A-Za-z]?\s+)?\d{{1,4}}(?:\s*{range_sep}\s*\d{{1,4}})?", line):
            keep = seen_table_header and previous_was_row
        elif previous_was_row and should_append_toc_continuation(line):
            keep = True

        if keep:
            filtered.append(line)
        previous_was_row = keep and not is_toc_header_line(line, toc_markers)
    return filtered


def parse_rule_based_toc_items(lines: list[str], default_source: str = "toc") -> list[dict]:
    items = []
    pending_item = None

    def append_title(target: dict, extra_text: str):
        extra = clean_toc_title(extra_text)
        if not extra:
            return
        target["title"] = clean_toc_title(f"{target['title']} {extra}".strip())
        target["displayTitle"] = target["title"]
        target["originalTitle"] = target["title"]

    def make_item(serial: str, title: str, page_from: int | None = None, page_to: int | None = None):
        clean_title = clean_toc_title(title)
        return {
            "serialNo": serial,
            "title": clean_title,
            "displayTitle": clean_title,
            "originalTitle": clean_title,
            "pageFrom": page_from,
            "pageTo": page_to,
            "courtFee": "",
            "source": default_source,
        }

    def finalize_pending_if_ready():
        nonlocal pending_item
        if pending_item and pending_item.get("title") and pending_item.get("pageFrom") is not None:
            items.append(pending_item)
            pending_item = None

    for line in lines:
        normalized_line = normalize_toc_parse_line(line)
        if len(normalized_line) < 1:
            continue
        if any(re.search(pattern, normalized_line.lower(), flags=re.IGNORECASE) for pattern in TOC_TABLE_HEADER_PATTERNS):
            continue
        if is_probable_stamp_noise(normalized_line):
            continue

        standalone_page_match = re.fullmatch(r"(?:[A-Za-z]{1,4}[-/]\d{1,4}[A-Za-z]?\s+)?(?P<from>\d{1,4})(?:\s*[\-–—]\s*(?P<to>\d{1,4}))?", normalized_line)
        if standalone_page_match and pending_item:
            page_from = int(standalone_page_match.group("from"))
            page_to = int(standalone_page_match.group("to") or page_from)
            if page_to < page_from:
                page_to = page_from
            pending_item["pageFrom"] = page_from
            pending_item["pageTo"] = page_to
            finalize_pending_if_ready()
            continue

        serial = ""
        body = normalized_line
        serial_match = re.match(r"^(?P<serial>\d{1,3})[\)\.\-: ]+(?P<body>.+)$", normalized_line)
        if serial_match:
            serial = serial_match.group("serial")
            body = serial_match.group("body").strip()

        title, page_from, page_to, annex = split_toc_page_suffix(body)
        if page_from is not None:
            if len(title) >= 3:
                finalize_pending_if_ready()
                items.append(make_item(serial, title, page_from=page_from, page_to=page_to))
                continue
            if pending_item:
                if annex and annex.lower() not in pending_item["title"].lower():
                    append_title(pending_item, annex)
                pending_item["pageFrom"] = page_from
                pending_item["pageTo"] = page_to
                finalize_pending_if_ready()
            continue

        if len(body) < 3:
            continue

        if serial:
            finalize_pending_if_ready()
            pending_item = make_item(serial, body)
            continue

        if pending_item and should_append_toc_continuation(body):
            append_title(pending_item, body)
            continue

        if items and should_append_toc_continuation(body) and len(body) <= 180:
            append_title(items[-1], body)

    finalize_pending_if_ready()
    return [
        item for item in items
        if item.get("title") and item.get("pageFrom") is not None and not is_toc_noise_line(item.get("title", ""))
    ]


def combine_toc_items(*groups: list[dict]) -> list[dict]:
    combined = []
    for group in groups:
        if isinstance(group, list):
            combined.extend(item for item in group if isinstance(item, dict))
    return combined


def finalize_extracted_page(page_num: int, page_data: dict) -> dict:
    text_value = page_data.get("text") or f"[Page {page_num} - no readable text detected]"
    extraction_method = page_data.get("extraction_method", "digital")
    source = "direct_text"
    if extraction_method == "ocr":
        source = "ocr"
    elif extraction_method == "vision_ocr":
        source = "vision"
    return {
        "page_num": page_num,
        "text": text_value,
        "used_ocr": bool(page_data.get("used_ocr")),
        "vision_used": bool(page_data.get("vision_used")),
        "handwriting_suspected": bool(page_data.get("handwriting_suspected")),
        "extraction_method": extraction_method,
        "source": source,
        "ocr_used": bool(page_data.get("used_ocr")),
        "char_count": len(text_value),
    }


def summarize_extracted_pages(pages_data: list[dict], total_pages: int, worker_count: int, pdf_id: str = "") -> dict:
    ocr_count = sum(1 for page in pages_data if page.get("used_ocr"))
    vision_ocr_count = sum(1 for page in pages_data if page.get("vision_used"))
    handwriting_count = sum(1 for page in pages_data if page.get("handwriting_suspected"))
    direct_text_count = sum(1 for page in pages_data if page.get("source") == "direct_text")
    stats = {
        "ocr_pages": ocr_count,
        "vision_ocr_pages": vision_ocr_count,
        "handwriting_suspected_pages": handwriting_count,
        "digital_pages": direct_text_count,
        "direct_text_pages": direct_text_count,
        "total_pages_processed": len(pages_data),
        "skipped_ocr_pages": len(pages_data) - ocr_count,
        "worker_count": worker_count,
    }
    log.info(
        "[VECTORIZE] pdf=%s extraction summary total_pages=%s workers=%s direct_text_pages=%s ocr_pages=%s skipped_ocr_pages=%s vision_pages=%s",
        pdf_id or "-",
        total_pages,
        worker_count,
        stats["direct_text_pages"],
        stats["ocr_pages"],
        stats["skipped_ocr_pages"],
        stats["vision_ocr_pages"],
    )
    return stats


def extract_pages_from_document(
    doc: fitz.Document,
    page_numbers: list[int],
    total_pages: int,
    dpi: int = 250,
    timing_collector: Optional[PdfTimingCollector] = None,
) -> tuple[list[dict], dict]:
    pages_data = []

    for page_num in page_numbers:
        page = doc[page_num - 1]
        page_data = extract_page_content(page, page_num, dpi=dpi, timing_collector=timing_collector)
        finalized = finalize_extracted_page(page_num, page_data)
        pages_data.append(finalized)
        log.info(
            "  Page %s/%s - %s%s - %s chars",
            page_num,
            total_pages,
            finalized["extraction_method"],
            " (handwriting assist)" if finalized["vision_used"] else "",
            finalized["char_count"],
        )

    stats = summarize_extracted_pages(pages_data, total_pages, worker_count=1)
    return pages_data, stats


def _extract_page_from_pdf_worker(pdf_path_str: str, page_num: int, total_pages: int, dpi: int = 250) -> dict:
    doc = fitz.open(pdf_path_str)
    try:
        page = doc[page_num - 1]
        page_data = extract_page_content(page, page_num, dpi=dpi, timing_collector=None)
    finally:
        doc.close()
    finalized = finalize_extracted_page(page_num, page_data)
    return {
        **finalized,
        "page_number": page_num,
    }


def extract_pages_from_pdf_parallel(
    pdf_path: Path,
    page_numbers: list[int],
    total_pages: int,
    dpi: int = 250,
    timing_collector: Optional[PdfTimingCollector] = None,
    worker_count: Optional[int] = None,
    pdf_id: str = "",
) -> tuple[list[dict], dict]:
    if not page_numbers:
        return [], summarize_extracted_pages([], total_pages, worker_count=0, pdf_id=pdf_id)

    effective_workers = max(1, min(int(worker_count or OCR_WORKER_COUNT), len(page_numbers)))
    log.info(
        "[VECTORIZE] pdf=%s starting parallel extraction total_pages=%s worker_count=%s dpi=%s",
        pdf_id or "-",
        len(page_numbers),
        effective_workers,
        dpi,
    )

    if effective_workers == 1:
        with fitz.open(pdf_path) as doc:
            pages_data, stats = extract_pages_from_document(doc, page_numbers, total_pages, dpi=dpi, timing_collector=timing_collector)
        stats["worker_count"] = 1
        return pages_data, stats

    started = time.perf_counter()
    pages_data = []
    try:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_map = {
                executor.submit(_extract_page_from_pdf_worker, str(pdf_path), page_num, total_pages, dpi): page_num
                for page_num in page_numbers
            }
            for future in as_completed(future_map):
                page_num = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.exception("[VECTORIZE] pdf=%s worker failed for page=%s; falling back to sequential extraction", pdf_id or "-", page_num)
                    raise exc
                pages_data.append({
                    "page_num": result["page_number"],
                    "text": result["text"],
                    "used_ocr": result["used_ocr"],
                    "vision_used": result["vision_used"],
                    "handwriting_suspected": result["handwriting_suspected"],
                    "extraction_method": result["extraction_method"],
                    "source": result["source"],
                    "ocr_used": result["ocr_used"],
                    "char_count": result["char_count"],
                })
    except Exception:
        with fitz.open(pdf_path) as doc:
            pages_data, stats = extract_pages_from_document(doc, page_numbers, total_pages, dpi=dpi, timing_collector=timing_collector)
        stats["worker_count"] = 1
        return pages_data, stats

    pages_data.sort(key=lambda item: item["page_num"])
    elapsed = time.perf_counter() - started
    stats = summarize_extracted_pages(pages_data, total_pages, worker_count=effective_workers, pdf_id=pdf_id)
    log.info(
        "[VECTORIZE] pdf=%s parallel extraction completed processed_pages=%s worker_count=%s extraction_time=%.3fs",
        pdf_id or "-",
        len(pages_data),
        effective_workers,
        elapsed,
    )
    return pages_data, stats


def build_vector_chunks(pdf_id: str, filename: str, pages_data: list[dict]) -> list[dict]:
    chunks = []
    for page in pages_data:
        chunks.append({
            "id": f"{pdf_id}_p{page['page_num']}",
            "text": page["text"],
            "metadata": {
                "page_num": page["page_num"],
                "used_ocr": page["used_ocr"],
                "vision_used": page["vision_used"],
                "handwriting_suspected": page["handwriting_suspected"],
                "extraction_method": page["extraction_method"],
                "filename": filename,
            },
        })
    return chunks


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

    vector_chunks = build_vector_chunks(pdf_id, filename, pages_data)
    total_chunks = len(vector_chunks)
    batch_size = VECTOR_DB_BATCH_SIZE
    log.info(
        "[VECTORIZE] pdf=%s total_chunks=%s db_batch_size=%s embedding_batch_size=%s embedding_device=%s embedding_model=%s",
        pdf_id,
        total_chunks,
        batch_size,
        EMBEDDING_BATCH_SIZE,
        get_embedder_device(),
        EMBEDDING_MODEL_NAME,
    )

    vectorization_scope = timing_collector.stage("total vectorization", "total_vectorization_time") if timing_collector else nullcontext()
    with vectorization_scope:
        chunk_started = time.perf_counter()
        if timing_collector:
            timing_collector.add_duration("chunking_time", time.perf_counter() - chunk_started)

        embedding_started = time.perf_counter()
        documents = [chunk["text"] for chunk in vector_chunks]
        embeddings = embed_texts(documents, batch_size=EMBEDDING_BATCH_SIZE, pdf_id=pdf_id)
        embedding_elapsed = time.perf_counter() - embedding_started
        log.info(
            "[VECTORIZE] pdf=%s embedding completed total_chunks=%s batch_size=%s device=%s total_time=%.3fs",
            pdf_id,
            total_chunks,
            EMBEDDING_BATCH_SIZE,
            get_embedder_device(),
            embedding_elapsed,
        )
        if timing_collector:
            timing_collector.add_duration("embedding_time", embedding_elapsed)

        total_batches = math.ceil(total_chunks / batch_size)
        total_db_insert_time = 0.0
        for batch_index in range(0, total_chunks, batch_size):
            batch = vector_chunks[batch_index:batch_index + batch_size]
            batch_embeddings = embeddings[batch_index:batch_index + batch_size]
            db_insert_started = time.perf_counter()
            collection.upsert(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
                embeddings=batch_embeddings,
            )
            elapsed = time.perf_counter() - db_insert_started
            total_db_insert_time += elapsed
            log.info(
                "[VECTORIZE] pdf=%s db batch %s/%s size=%s took %.3fs",
                pdf_id,
                (batch_index // batch_size) + 1,
                total_batches,
                len(batch),
                elapsed,
            )
            if timing_collector:
                timing_collector.add_duration("vector_db_insert_time", elapsed)
        log.info(
            "[VECTORIZE] pdf=%s chroma upsert completed total_batches=%s total_insert_time=%.3fs",
            pdf_id,
            total_batches,
            total_db_insert_time,
        )
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
        if item.get("source") == "gap":
            continue
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
            if item.get("pdfPageTo") is not None:
                prev["pdfPageTo"] = item.get("pdfPageTo")
            if item.get("tocPageTo") is not None:
                prev["tocPageTo"] = item.get("tocPageTo")
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
        if item.get("source") == "gap":
            classified.append(dict(item))
            continue
        raw_title = item.get("title", "")
        preview = build_segment_preview(all_pages, item.get("pageFrom", 1), item.get("pageTo", 1))
        direct = direct_parent_match(raw_title, preview)
        if direct:
            scored = [(10.0, direct)]
            candidates = [direct]
            chosen_title = direct
        else:
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


class IndexSaveRequest(BaseModel):
    index: list[dict]


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
    refreshed = refresh_saved_index_if_needed(pdf_id, record=record, saved_index=saved_index, reason="loading saved PDF")
    if refreshed:
        record = get_pdf_record(pdf_id) or record
        saved_index = refreshed.get("index", saved_index)
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
    existing_record = get_pdf_record(pdf_id) or {}
    pdf_path = stored_pdf_path(pdf_id)
    pdf_path.write_bytes(pdf_bytes)
    log.info("Full-document ingest for PDF: %s  id=%s", filename, pdf_id)
    timing_collector = PdfTimingCollector(pdf_id, filename)

    with timing_collector.stage("file open", "file_open"):
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = doc.page_count
    if total_pages < 1:
        doc.close()
        raise HTTPException(400, "PDF has no pages")

    try:
        selected_page_numbers = list(range(1, total_pages + 1))
    finally:
        doc.close()

    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=filename,
        cnr_number=cnr_number_from_filename(filename),
        file_size_bytes=len(pdf_bytes),
        total_pages=total_pages,
        selected_start_page=1,
        selected_end_page=total_pages,
        indexed_pages=0,
        status="extracting_text",
        retrieval_status="full_ingestion_running",
        index_ready=False,
        chat_ready=False,
        pending_pages=total_pages,
        index_source="",
        queue_bucket="index",
        deferred_decision="completed",
        last_error="",
        review_reason="",
        batch_run_id=existing_record.get("batch_run_id", ""),
        batch_enqueued_at=existing_record.get("batch_enqueued_at", ""),
    )

    with timing_collector.stage("full text extraction", "full_text_extraction_time"):
        pages_data, stats = extract_pages_from_pdf_parallel(
            pdf_path,
            selected_page_numbers,
            total_pages,
            dpi=250,
            timing_collector=timing_collector,
            worker_count=OCR_WORKER_COUNT,
            pdf_id=pdf_id,
        )

    update_pdf_record(
        pdf_id,
        status="vectorizing",
        retrieval_status="full_ingestion_running",
        indexed_pages=len(pages_data),
        pending_pages=0,
        selected_start_page=1,
        selected_end_page=total_pages,
    )
    replace_extracted_pages(pdf_id, pages_data, stage="full_document_ingestion")
    upsert_collection_pages(pdf_id, filename, pages_data, reset=True, timing_collector=timing_collector)

    indexed_pages = total_pages
    pending_pages = 0
    cnr_number = cnr_number_from_filename(filename)
    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=filename,
        cnr_number=cnr_number,
        file_size_bytes=len(pdf_bytes),
        total_pages=total_pages,
        selected_start_page=1,
        selected_end_page=total_pages,
        indexed_pages=indexed_pages,
        status="vectorized",
        retrieval_status="vectorized",
        index_ready=False,
        chat_ready=True,
        pending_pages=pending_pages,
        index_source="",
        queue_bucket="library",
        deferred_decision="completed",
        last_error="",
        review_reason="",
        batch_run_id=existing_record.get("batch_run_id", ""),
        batch_enqueued_at=existing_record.get("batch_enqueued_at", ""),
    )

    update_batch_report_for_pdf(pdf_id)
    timing_collector.log_summary("full_document_ingest")

    return {
        "pdf_id": pdf_id,
        "cnr_number": cnr_number,
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
        "pending_pages": pending_pages,
        "chat_ready": True,
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
    update_pdf_record(
        ingest_resp["pdf_id"],
        status="building_index",
        retrieval_status="vectorized",
        indexed_pages=ingest_resp.get("indexed_pages", 0),
        pending_pages=0,
    )
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


def enqueue_pdf_for_stage1(
    pdf_bytes: bytes,
    filename: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    batch_run_id: str = "",
    batch_enqueued_at: str = "",
) -> dict:
    pdf_id = pdf_id_from_bytes(pdf_bytes)
    existing = build_existing_pdf_payload(pdf_id, filename)
    if existing:
        return {
            "pdf_id": pdf_id,
            "filename": existing.get("filename") or filename,
            "status": "skipped",
            "skipped_duplicate": True,
            "reason": existing.get("message") or "PDF already exists",
        }

    pdf_path = stored_pdf_path(pdf_id)
    pdf_path.write_bytes(pdf_bytes)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
    if total_pages < 1:
        raise HTTPException(400, "PDF has no pages")

    start_page = max(1, min(int(start_page or 1), total_pages))
    default_end_page = min(total_pages, start_page + 9)
    end_page = default_end_page if end_page is None else max(start_page, min(int(end_page), total_pages))
    cnr_number = cnr_number_from_filename(filename)
    upsert_pdf_record(
        pdf_id=pdf_id,
        filename=filename,
        cnr_number=cnr_number,
        file_size_bytes=len(pdf_bytes),
        total_pages=total_pages,
        selected_start_page=start_page,
        selected_end_page=end_page,
        indexed_pages=0,
        status="queued_for_stage1",
        retrieval_status="queued_for_stage1",
        index_ready=False,
        chat_ready=False,
        pending_pages=total_pages,
        index_source="",
        queue_bucket="stage1_batch",
        deferred_decision="queue",
        last_error="",
        review_reason="",
        batch_run_id=batch_run_id,
        batch_enqueued_at=batch_enqueued_at,
    )
    update_batch_report_for_pdf(pdf_id)
    return {
        "pdf_id": pdf_id,
        "filename": filename,
        "cnr_number": cnr_number,
        "total_pages": total_pages,
        "status": "queued_for_stage1",
        "retrieval_status": "queued_for_stage1",
        "pending_pages": total_pages,
        "queued": True,
        "skipped_duplicate": False,
        "batch_run_id": batch_run_id,
    }


def normalize_background_queue_state():
    for record in list_pdf_records():
        pdf_id = record.get("pdf_id")
        if not pdf_id:
            continue

        status = str(record.get("status") or "").lower()
        retrieval_status = str(record.get("retrieval_status") or "").lower()
        queue_bucket = str(record.get("queue_bucket") or "")
        pending_pages = int(record.get("pending_pages") or 0)
        chat_ready = bool(record.get("chat_ready"))

        if pending_pages <= 0 and (chat_ready or retrieval_status == "vectorized" or status == "vectorized"):
            if queue_bucket != "library" or status != "vectorized" or retrieval_status != "vectorized":
                update_pdf_record(
                    pdf_id,
                    status="vectorized",
                    retrieval_status="vectorized",
                    chat_ready=True,
                    pending_pages=0,
                    queue_bucket="library",
                    deferred_decision="completed",
                )
            continue

        if queue_bucket == "stage1_batch" and status == "index_ready":
            update_pdf_record(
                pdf_id,
                queue_bucket="deferred",
                retrieval_status="queued_for_full_ingestion",
                deferred_decision="pending",
                last_error="",
            )
            continue

        if queue_bucket == "stage1_batch" and (status == "full_ingestion_running" or retrieval_status == "full_ingestion_running"):
            update_pdf_record(
                pdf_id,
                queue_bucket="deferred",
                status="full_ingestion_running",
                retrieval_status="full_ingestion_running",
                deferred_decision="pending",
            )


def process_stage1_batch_pdf_impl(pdf_id: str) -> dict:
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    pdf_path = stored_pdf_path(pdf_id)
    if not pdf_path.exists():
        raise HTTPException(404, f"Stored PDF for {pdf_id} not found")

    filename = record.get("filename") or f"{pdf_id}.pdf"
    start_page = int(record.get("selected_start_page") or 1)
    end_page = int(record.get("selected_end_page") or min(int(record.get("total_pages") or 1), start_page + 9))
    last_error = None

    for attempt in range(1, 4):
        try:
            update_pdf_record(
                pdf_id,
                status="indexing_running",
                retrieval_status="queued_for_stage1",
                queue_bucket="stage1_batch",
                deferred_decision="queue",
                last_error="",
            )
            payload = build_stage_one_payload(pdf_path.read_bytes(), filename, start_page=start_page, end_page=end_page)
            refreshed_record = get_pdf_record(pdf_id) or record
            if int(payload.get("pending_pages") or 0) > 0:
                update_pdf_record(
                    pdf_id,
                    status="index_ready",
                    retrieval_status="queued_for_full_ingestion",
                    queue_bucket="deferred",
                    deferred_decision="pending",
                    last_error="",
                )
                start_deferred_runner_if_needed()
                refreshed_record = get_pdf_record(pdf_id) or refreshed_record
                return {
                    **payload,
                    "status": refreshed_record.get("status", payload.get("status", "index_ready")),
                    "retrieval_status": refreshed_record.get("retrieval_status", payload.get("retrieval_status", "queued_for_full_ingestion")),
                    "pending_pages": refreshed_record.get("pending_pages", payload.get("pending_pages", 0)),
                    "chat_ready": bool(refreshed_record.get("chat_ready", payload.get("chat_ready", False))),
                }

            update_pdf_record(
                pdf_id,
                status="vectorized",
                retrieval_status="vectorized",
                chat_ready=True,
                pending_pages=0,
                queue_bucket="library",
                deferred_decision="completed",
                last_error="",
                review_reason="",
            )
            return payload
        except Exception as exc:
            last_error = str(exc)
            log.exception("Batch indexing worker failed for %s on attempt %s", pdf_id, attempt)
            if attempt < 3:
                update_pdf_record(
                    pdf_id,
                    status="queued_for_stage1",
                    retrieval_status="queued_for_stage1",
                    queue_bucket="stage1_batch",
                    deferred_decision="queue",
                    last_error=last_error,
                )
                time.sleep(min(6, attempt * 2))
                continue
            update_pdf_record(
                pdf_id,
                status="failed",
                retrieval_status="failed",
                queue_bucket="stage1_batch",
                deferred_decision="queue",
                last_error=last_error,
            )
            raise

    raise RuntimeError(last_error or f"Batch indexing worker failed for {pdf_id}")


def run_stage1_batch_queue_worker():
    stage1_batch_runner_status.update({
        "running": True,
        "processed": 0,
        "total": len(list_stage1_batch_pdf_ids()),
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": time.time(),
        "started_at": time.time(),
        "status": "running",
        "stop_requested": False,
    })
    processed = 0
    try:
        while True:
            if stage1_batch_runner_status.get("stop_requested"):
                stage1_batch_runner_status.update({
                    "status": "stopping",
                    "heartbeat_ts": time.time(),
                })
                break
            pending_ids = list_stage1_batch_pdf_ids()
            if not pending_ids:
                break
            pdf_id = pending_ids[0]
            record = get_pdf_record(pdf_id) or {}
            stage1_batch_runner_status.update({
                "total": processed + len(pending_ids),
                "current_pdf_id": pdf_id,
                "current_filename": record.get("filename", ""),
                "processed": processed,
                "heartbeat_ts": time.time(),
            })
            try:
                process_stage1_batch_pdf_impl(pdf_id)
                processed += 1
            except Exception as exc:
                stage1_batch_runner_status["last_error"] = str(exc)
                processed += 1
            finally:
                stage1_batch_runner_status["processed"] = processed
                stage1_batch_runner_status["heartbeat_ts"] = time.time()
    finally:
        stage1_batch_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
            "status": "idle",
            "stop_requested": False,
        })


def start_stage1_batch_runner_if_needed() -> bool:
    with stage1_batch_runner_lock:
        if stage1_batch_runner_status.get("running"):
            return False
        if not list_stage1_batch_pdf_ids():
            return False
        worker = Thread(target=run_stage1_batch_queue_worker, daemon=True)
        worker.start()
        return True


def analyze_saved_index_health(index_items: list[dict], total_pages: int) -> dict:
    gap_items = [item for item in index_items if item.get("source") == "gap"]
    non_gap_items = [item for item in index_items if item.get("source") != "gap"]
    auto_items = [item for item in non_gap_items if str(item.get("source") or "").strip() == "auto"]
    gap_pages = sum(
        max(0, coerce_page_number(item.get("pageTo"), 0) - coerce_page_number(item.get("pageFrom"), 0) + 1)
        for item in gap_items
    )
    largest_gap = max(
        (
            max(0, coerce_page_number(item.get("pageTo"), 0) - coerce_page_number(item.get("pageFrom"), 0) + 1)
            for item in gap_items
        ),
        default=0,
    )

    reasons = []
    if total_pages >= 20 and not non_gap_items:
        reasons.append("no_named_sections")
    if total_pages >= 20 and len(non_gap_items) <= 1 and gap_pages >= max(total_pages - 1, 1):
        reasons.append("single_section_with_near_total_gap")
    if total_pages >= 30 and len(non_gap_items) <= 2 and gap_pages / max(total_pages, 1) >= 0.80:
        reasons.append("gap_dominates_document")
    if total_pages >= 30 and largest_gap / max(total_pages, 1) >= 0.75:
        reasons.append("very_large_gap")
    if total_pages >= 6 and len(non_gap_items) <= 1 and auto_items:
        reasons.append("single_ai_section_for_multi_page_pdf")

    return {
        "suspicious": bool(reasons),
        "reasons": reasons,
    }


def should_refresh_saved_index(record: Optional[dict], saved_index: list[dict]) -> tuple[bool, list[str]]:
    if not record:
        return False, []

    total_pages = int(record.get("total_pages") or 0)
    indexed_pages = int(record.get("indexed_pages") or 0)
    pending_pages = int(record.get("pending_pages") or 0)
    retrieval_status = (record.get("retrieval_status") or "").strip().lower()
    fully_vectorized = bool(total_pages) and indexed_pages >= total_pages and pending_pages == 0 and retrieval_status == "vectorized"
    if not fully_vectorized:
        return False, []

    reasons = []
    if not saved_index:
        reasons.append("missing_saved_index")

    selected_start_page = int(record.get("selected_start_page") or 1)
    selected_end_page = int(record.get("selected_end_page") or 0)
    if selected_start_page != 1 or selected_end_page < total_pages:
        reasons.append("stale_index_range_metadata")

    if saved_index:
        health = analyze_saved_index_health(saved_index, total_pages)
        if health["suspicious"]:
            reasons.extend(health["reasons"])

    reasons = list(dict.fromkeys(reasons))
    return bool(reasons), reasons


def refresh_saved_index_if_needed(
    pdf_id: str,
    record: Optional[dict] = None,
    saved_index: Optional[list[dict]] = None,
    reason: str = "",
) -> Optional[dict]:
    record = record or get_pdf_record(pdf_id)
    if not record:
        return None

    saved_index = saved_index if saved_index is not None else get_saved_index(pdf_id)
    needs_refresh, reasons = should_refresh_saved_index(record, saved_index)
    if not needs_refresh:
        return None

    log.info(
        "Refreshing saved index for %s after %s because %s",
        pdf_id,
        reason or "validation",
        ", ".join(reasons),
    )
    payload = generate_index_payload(IndexRequest(pdf_id=pdf_id))
    refreshed_record = get_pdf_record(pdf_id) or record
    total_pages = int(refreshed_record.get("total_pages") or payload.get("total_pages") or 0)
    update_pdf_record(
        pdf_id,
        status="vectorized",
        retrieval_status="vectorized",
        chat_ready=True,
        pending_pages=0,
        queue_bucket="library",
        deferred_decision="completed",
        last_error="",
        index_ready=True,
        index_source=payload.get("index_source", refreshed_record.get("index_source", "auto")),
        indexed_pages=max(int(payload.get("indexed_pages") or 0), total_pages),
        selected_start_page=1,
        selected_end_page=total_pages or int(payload.get("indexed_page_end") or refreshed_record.get("selected_end_page") or 1),
    )
    return payload


INDEX_REVIEW_REASON_LABELS = {
    "missing_saved_index": "Saved index is missing",
    "stale_index_range_metadata": "Indexed page range does not cover the full PDF",
    "no_named_sections": "No named index sections were found",
    "single_section_with_near_total_gap": "Only one named section was found and the rest is a large gap",
    "gap_dominates_document": "Most of the index is covered by generic gaps",
    "very_large_gap": "The index contains an unusually large gap range",
    "explicit_toc_detected_but_not_extracted": "A visible TOC/index page was detected but usable TOC rows were not extracted",
    "toc_detected_but_final_index_used_ai_fallback": "A visible TOC/index page was detected but AI fallback was used in the final index",
    "single_ai_section_for_multi_page_pdf": "Only one broad AI section was created for a multi-page PDF",
    "too_few_sections_for_visible_toc": "Too few final sections were created despite a visible TOC/index page",
}


def format_review_reason(reason_codes: list[str]) -> str:
    labels = [INDEX_REVIEW_REASON_LABELS.get(code, code.replace("_", " ").strip()) for code in reason_codes if code]
    labels = list(dict.fromkeys(labels))
    return "; ".join(labels)[:500]


def is_fully_vectorized_record(record: Optional[dict]) -> bool:
    if not record:
        return False
    total_pages = int(record.get("total_pages") or 0)
    indexed_pages = int(record.get("indexed_pages") or 0)
    pending_pages = int(record.get("pending_pages") or 0)
    retrieval_status = (record.get("retrieval_status") or "").strip().lower()
    return bool(total_pages) and indexed_pages >= total_pages and pending_pages == 0 and retrieval_status == "vectorized"


def filter_pdf_records_for_audit(search: str = "", batch_filter: str = "", row_start: int = 1, row_end: Optional[int] = None) -> list[dict]:
    records = [record for record in list_pdf_records(search) if is_fully_vectorized_record(record)]
    batch_value = (batch_filter or "").strip().lower()
    if batch_value:
        records = [
            record for record in records
            if batch_value in str(record.get("cnr_number") or "").lower()
            or batch_value in str(record.get("filename") or "").lower()
        ]

    start_index = max(int(row_start or 1), 1) - 1
    end_index = None if row_end in (None, "", 0, "0") else max(int(row_end), start_index + 1)
    if start_index >= len(records):
        return []
    return records[start_index:end_index]


def audit_saved_index_record(pdf_id: str, record: Optional[dict] = None, saved_index: Optional[list[dict]] = None) -> dict:
    record = record or get_pdf_record(pdf_id)
    if not is_fully_vectorized_record(record):
        return {"pdf_id": pdf_id, "eligible": False, "suspicious": False, "reasons": []}

    saved_index = saved_index if saved_index is not None else get_saved_index(pdf_id)
    suspicious, reason_codes = should_refresh_saved_index(record, saved_index)
    review_reason = format_review_reason(reason_codes)
    if suspicious:
        update_pdf_record(
            pdf_id,
            status="needs_review",
            queue_bucket="reindex_review",
            review_reason=review_reason,
            index_ready=True,
        )
    else:
        update_pdf_record(
            pdf_id,
            status="vectorized",
            queue_bucket="library",
            review_reason="",
            index_ready=True,
        )

    return {
        "pdf_id": pdf_id,
        "eligible": True,
        "suspicious": suspicious,
        "reasons": reason_codes,
        "review_reason": review_reason,
    }


def run_index_audit_worker(records: list[dict]):
    audit_runner_status.update({
        "running": True,
        "processed": 0,
        "total": len(records),
        "flagged": 0,
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": time.time(),
        "started_at": time.time(),
        "status": "running",
        "stop_requested": False,
    })
    flagged = 0
    try:
        for index, record in enumerate(records, start=1):
            if audit_runner_status.get("stop_requested"):
                audit_runner_status.update({"status": "stopping", "heartbeat_ts": time.time()})
                break
            audit_runner_status.update({
                "current_pdf_id": record.get("pdf_id", ""),
                "current_filename": record.get("filename", ""),
                "processed": index - 1,
                "heartbeat_ts": time.time(),
            })
            try:
                result = audit_saved_index_record(record["pdf_id"], record=record)
                if result.get("suspicious"):
                    flagged += 1
                    audit_runner_status["flagged"] = flagged
            except Exception as exc:
                log.exception("Index audit failed for %s", record.get("pdf_id"))
                update_pdf_record(record["pdf_id"], last_error=str(exc))
                audit_runner_status["last_error"] = str(exc)
            finally:
                audit_runner_status["processed"] = index
                audit_runner_status["heartbeat_ts"] = time.time()
    finally:
        audit_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
            "status": "idle",
            "stop_requested": False,
        })


def run_reindex_review_worker(pdf_ids: list[str]):
    reindex_review_runner_status.update({
        "running": True,
        "processed": 0,
        "total": len(pdf_ids),
        "fixed": 0,
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": time.time(),
        "started_at": time.time(),
        "status": "running",
        "stop_requested": False,
    })
    fixed = 0
    try:
        for index, pdf_id in enumerate(pdf_ids, start=1):
            if reindex_review_runner_status.get("stop_requested"):
                reindex_review_runner_status.update({"status": "stopping", "heartbeat_ts": time.time()})
                break
            record = get_pdf_record(pdf_id) or {}
            reindex_review_runner_status.update({
                "current_pdf_id": pdf_id,
                "current_filename": record.get("filename", ""),
                "processed": index - 1,
                "heartbeat_ts": time.time(),
            })
            try:
                payload = generate_index_payload(IndexRequest(pdf_id=pdf_id))
                refreshed = get_pdf_record(pdf_id) or record
                total_pages = int(refreshed.get("total_pages") or payload.get("total_pages") or 0)
                update_pdf_record(
                    pdf_id,
                    status="vectorized",
                    retrieval_status="vectorized",
                    chat_ready=True,
                    pending_pages=0,
                    queue_bucket="library",
                    deferred_decision="completed",
                    last_error="",
                    review_reason="",
                    index_ready=True,
                    index_source=payload.get("index_source", refreshed.get("index_source", "auto")),
                    indexed_pages=max(int(payload.get("indexed_pages") or 0), total_pages),
                    selected_start_page=1,
                    selected_end_page=total_pages or int(payload.get("indexed_page_end") or refreshed.get("selected_end_page") or 1),
                )
                fixed += 1
                reindex_review_runner_status["fixed"] = fixed
            except Exception as exc:
                log.exception("Reindex review queue failed for %s", pdf_id)
                update_pdf_record(pdf_id, status="needs_review", queue_bucket="reindex_review", last_error=str(exc))
                reindex_review_runner_status["last_error"] = str(exc)
            finally:
                reindex_review_runner_status["processed"] = index
                reindex_review_runner_status["heartbeat_ts"] = time.time()
    finally:
        reindex_review_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
            "status": "idle",
            "stop_requested": False,
        })


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
        "stop_requested": False,
    })
    try:
        if index_runner_status.get("stop_requested"):
            index_runner_status.update({
                "running": False,
                "current_pdf_id": "",
                "current_filename": "",
                "heartbeat_ts": time.time(),
                "finished_pdf_id": pdf_id,
                "finished_filename": filename,
                "status": "stopped",
                "stop_requested": False,
            })
            return
        build_stage_one_payload(pdf_bytes, filename, start_page=start_page, end_page=end_page)
        index_runner_status.update({
            "running": False,
            "current_pdf_id": "",
            "current_filename": "",
            "heartbeat_ts": time.time(),
            "finished_pdf_id": pdf_id,
            "finished_filename": filename,
            "status": "completed",
            "stop_requested": False,
        })
    except Exception as exc:
        log.exception("Background indexing failed for %s", filename)
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
            "stop_requested": False,
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
    Accuracy-first indexing pipeline.
    Save the PDF, extract the whole document, vectorize the full text,
    then generate the final index from full-document evidence.
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
    Re-run full-document indexing for an already-saved PDF without requiring the browser
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
def generate_index_payload(req: IndexRequest, timing_collector: Optional[PdfTimingCollector] = None) -> dict:
    """
    Generate the structured document index from the full cached document text.
    If a TOC is detected, ranges are expanded deterministically across the whole PDF.
    """
    record = get_pdf_record(req.pdf_id)
    timing_collector = timing_collector or PdfTimingCollector(req.pdf_id, (record or {}).get("filename", ""))
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
    pdf_path = stored_pdf_path(req.pdf_id)
    toc_llm_calls = 0

    def read_direct_toc_text_map(candidate_pages: list[dict]) -> dict[int, str]:
        direct_map = {page["page_num"]: "" for page in candidate_pages}
        if not candidate_pages or not pdf_path.exists():
            return direct_map

        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            log.warning("Could not open stored PDF for direct TOC text parsing: %s", exc)
            return direct_map

        try:
            for page in candidate_pages:
                page_num = page["page_num"]
                if 1 <= page_num <= doc.page_count:
                    direct_map[page_num] = normalize_ocr_text(doc[page_num - 1].get_text("text"))
        finally:
            doc.close()
        return direct_map

    def run_toc_extraction(
        candidate_pages: list[dict],
        allow_image_fallback: bool = False,
        allow_text_llm: bool = True,
    ) -> dict:
        nonlocal toc_llm_calls
        result = {
            "page_nums": [],
            "rule_items": [],
            "text_items": [],
            "image_items": [],
            "accepted_items": [],
            "accepted_source": "",
            "candidate_line_count": 0,
            "ocr_pages": [],
            "skipped_pages": [],
            "force_image_retry": False,
        }
        if not candidate_pages:
            return result

        direct_text_map = read_direct_toc_text_map(candidate_pages)
        filtered_lines: list[str] = []

        for page in candidate_pages:
            page_num = page["page_num"]
            direct_text = direct_text_map.get(page_num, "")
            selected_text = page.get("text", "") or direct_text
            toc_block_text = isolate_toc_text_block(selected_text)
            used_ocr_for_toc = page.get("used_ocr", False) or needs_ocr(direct_text)
            result["page_nums"].append(page_num)
            if used_ocr_for_toc:
                result["ocr_pages"].append(page_num)
                debug_dump(f"TOC OCR text page {page_num}", selected_text)
            else:
                result["skipped_pages"].append(page_num)
            debug_dump(f"TOC direct text page {page_num}", direct_text)
            debug_dump(f"TOC isolated block page {page_num}", toc_block_text)
            filtered_lines.extend(filter_toc_lines(toc_block_text, toc_markers))

        deduped_lines = []
        seen_lines = set()
        for line in filtered_lines:
            key = line.lower()
            if key in seen_lines:
                continue
            deduped_lines.append(line)
            seen_lines.add(key)
        filtered_lines = deduped_lines
        result["candidate_line_count"] = len(filtered_lines)

        log.info("TOC OCR pages for %s: used=%s skipped=%s", req.pdf_id, result["ocr_pages"], result["skipped_pages"])
        log.info("TOC candidate line count for %s: %s", req.pdf_id, result["candidate_line_count"])
        debug_dump("TOC filtered lines", "\n".join(filtered_lines))

        rule_quality = None
        result["rule_items"] = parse_rule_based_toc_items(filtered_lines, default_source="toc")
        if result["rule_items"]:
            log.info("Rule-based TOC parsing found %s rows from pages %s", len(result["rule_items"]), result["page_nums"])
            rule_quality = evaluate_toc_items_confidence(result["rule_items"], toc_range_end)
            log.info(
                "TOC rule quality for %s: kept=%s/%s coverage=%.2f ascending=%.2f rejected=%s",
                req.pdf_id,
                rule_quality["kept_items"],
                rule_quality["total_items"],
                rule_quality["coverage_ratio"],
                rule_quality["ascending_ratio"],
                rule_quality["rejected_titles"],
            )
            if rule_quality["accepted"] and toc_acceptance_floor(rule_quality, candidate_pages):
                result["accepted_items"] = rule_quality["items"]
                result["accepted_source"] = "toc"

        result["force_image_retry"] = should_force_image_toc_retry(candidate_pages, result, toc_markers, rule_quality)
        if result["force_image_retry"]:
            log.info("Forcing TOC image retry for %s on pages %s after weak/noisy scanned TOC parse", req.pdf_id, result["page_nums"])

        if allow_text_llm and not result["accepted_items"] and filtered_lines and not result["force_image_retry"]:
            toc_prompt = f"""You are reading filtered TOC candidate lines from an Indian court document.
These lines may come from a real Table of Contents / Index spanning multiple pages.

Task:
- Decide whether these lines represent a real TOC/index table.
- If yes, merge continuation lines into the correct row and return strict JSON only.
- If not, return [].
- Preserve Hindi and English exactly as written.
- Convert Hindi digits to Arabic numerals in pageFrom/pageTo.
- Keep courtFee empty if unavailable.
- Do not add commentary.

Candidate pages: {result['page_nums']}
Filtered lines:
{chr(10).join(filtered_lines[:220])}

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
            debug_dump("TOC raw prompt", toc_prompt)
            toc_llm_calls += 1
            toc_raw = call_local_text(
                messages=[{"role": "user", "content": toc_prompt}],
                max_tokens=2200,
            )
            debug_dump("TOC raw response", toc_raw)
            result["text_items"] = extract_json_list(toc_raw)
            for row in result["text_items"]:
                row.setdefault("source", "toc")
            if result["text_items"]:
                log.info("Merged TOC LLM call found %s rows from pages %s", len(result["text_items"]), result["page_nums"])
                text_quality = evaluate_toc_items_confidence(result["text_items"], toc_range_end)
                log.info(
                    "TOC text quality for %s: kept=%s/%s coverage=%.2f ascending=%.2f rejected=%s",
                    req.pdf_id,
                    text_quality["kept_items"],
                    text_quality["total_items"],
                    text_quality["coverage_ratio"],
                    text_quality["ascending_ratio"],
                    text_quality["rejected_titles"],
                )
                if text_quality["accepted"] and toc_acceptance_floor(text_quality, candidate_pages):
                    result["accepted_items"] = text_quality["items"]
                    result["accepted_source"] = "toc"
            elif toc_raw.strip():
                log.info("TOC text response for pages %s was not usable JSON (%s chars)", result["page_nums"], len(toc_raw))

        if not result["accepted_items"] and allow_image_fallback:
            result["image_items"] = extract_toc_from_page_images(pdf_path, result["page_nums"])
            if result["image_items"]:
                log.info("TOC image fallback found %s rows from pages %s", len(result["image_items"]), result["page_nums"])
                image_quality = evaluate_toc_items_confidence(result["image_items"], toc_range_end)
                log.info(
                    "TOC image quality for %s: kept=%s/%s coverage=%.2f ascending=%.2f rejected=%s",
                    req.pdf_id,
                    image_quality["kept_items"],
                    image_quality["total_items"],
                    image_quality["coverage_ratio"],
                    image_quality["ascending_ratio"],
                    image_quality["rejected_titles"],
                )
                if image_quality["accepted"] and toc_acceptance_floor(image_quality, candidate_pages):
                    result["accepted_items"] = image_quality["items"]
                    result["accepted_source"] = "toc-image"

        return result

    index_items = []
    toc_source = ""
    toc_range_end = total_pages if record else indexed_end
    toc_page_nums = []
    toc_rule_items = []
    toc_image_items = []
    toc_text_items = []
    explicit_toc_detected = False

    with timing_collector.stage("TOC detection", "toc_detection"):
        toc_started = time.perf_counter()
        primary_candidates = collect_toc_candidate_pages(all_pages, toc_markers, max_pages=6)
        fallback_candidates = collect_toc_fallback_pages(all_pages, toc_markers, max_pages=6)
        ranked_primary = rank_toc_candidate_pages(primary_candidates, toc_markers)
        ranked_fallback = rank_toc_candidate_pages(fallback_candidates, toc_markers)
        all_candidate_pages = []
        seen_candidate_pages = set()
        for ranked_group in (ranked_primary, ranked_fallback):
            for item in ranked_group:
                page_num = item["page"]["page_num"]
                if page_num in seen_candidate_pages:
                    continue
                seen_candidate_pages.add(page_num)
                all_candidate_pages.append(item)
        log.info(
            "TOC candidate pages for %s: %s",
            req.pdf_id,
            [
                {
                    "page": item["page"]["page_num"],
                    "score": item["score"],
                    "explicit": item.get("explicit", False),
                }
                for item in all_candidate_pages
            ],
        )
        explicit_toc_detected = any(item.get("explicit") for item in all_candidate_pages)

        selected_pages = select_toc_pages_for_extraction(all_candidate_pages, max_pages=3)
        fallback_selected_pages = select_toc_pages_for_extraction(ranked_fallback, max_pages=3)
        fallback_page_nums = [page["page_num"] for page in fallback_selected_pages]
        log.info("TOC selected pages for %s: primary=%s fallback=%s", req.pdf_id, [page["page_num"] for page in selected_pages], fallback_page_nums)

        first_text_pages = selected_pages or fallback_selected_pages
        first_pass = run_toc_extraction(first_text_pages, allow_image_fallback=False, allow_text_llm=True)
        toc_page_nums = first_pass["page_nums"]
        toc_rule_items = first_pass["rule_items"]
        toc_text_items = first_pass["text_items"]
        toc_image_items = first_pass["image_items"]
        combined_toc_items = first_pass["accepted_items"]
        toc_source = first_pass["accepted_source"]
        candidate_line_count = first_pass["candidate_line_count"]
        force_image_retry = bool(first_pass.get("force_image_retry"))

        image_fallback_pages = []
        if fallback_selected_pages and fallback_page_nums != toc_page_nums:
            image_fallback_pages = fallback_selected_pages
        elif selected_pages:
            image_fallback_pages = selected_pages

        if force_image_retry:
            image_fallback_pages = first_text_pages or image_fallback_pages

        if (not combined_toc_items or force_image_retry) and image_fallback_pages:
            log.info("Retrying TOC extraction with image fallback pages for %s: %s", req.pdf_id, [page["page_num"] for page in image_fallback_pages])
            image_pass = run_toc_extraction(image_fallback_pages, allow_image_fallback=True, allow_text_llm=False)
            toc_page_nums = image_pass["page_nums"]
            toc_rule_items = image_pass["rule_items"]
            toc_text_items = image_pass["text_items"]
            toc_image_items = image_pass["image_items"]
            if image_pass["accepted_items"]:
                combined_toc_items = image_pass["accepted_items"]
            toc_source = image_pass["accepted_source"] or toc_source
            candidate_line_count = image_pass["candidate_line_count"]

        if combined_toc_items:
            index_items = build_toc_ranges_from_items(combined_toc_items, indexed_start, toc_range_end, "toc", toc_page_nums=toc_page_nums)
            toc_source = toc_source or ("toc-image" if toc_image_items and not (toc_rule_items or toc_text_items) else "toc")
            log.info("TOC merged across pages %s into %s unique rows", toc_page_nums, len(index_items))
            debug_dump("TOC final rows", index_items)

        toc_elapsed = time.perf_counter() - toc_started
        log.info(
            "TOC detection stats for %s: candidate_line_count=%s toc_llm_calls=%s toc_detection_time=%.3fs",
            req.pdf_id,
            candidate_line_count,
            toc_llm_calls,
            toc_elapsed,
        )

    with timing_collector.stage("LLM indexing", "llm_indexing_time"):

        scan_chunk = 25
        auto_items = []
        uncovered_pages = all_pages if (not index_items and not explicit_toc_detected) else []
        if not index_items and explicit_toc_detected:
            log.warning("Visible TOC detected for %s but no reliable TOC rows were extracted. Routing to review instead of trusting AI fallback.", req.pdf_id)
        elif not index_items:
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
                    "pdfPageFrom": cursor,
                    "pdfPageTo": pf - 1,
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
                    "pdfPageFrom": item.get("pdfPageFrom", pf),
                    "pdfPageTo": item.get("pdfPageTo", pt),
                    "tocPageFrom": item.get("tocPageFrom"),
                    "tocPageTo": item.get("tocPageTo"),
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
                "pdfPageFrom": cursor,
                "pdfPageTo": final_range_end,
                "source": "gap",
                "serialNo": "",
                "courtFee": "",
            })

        classified_final = classify_index_to_parent_documents(final, all_pages)
        strict_review_reasons = build_strict_index_review_reasons(total_pages, explicit_toc_detected, index_items, classified_final)
        review_reason = format_review_reason(strict_review_reasons)
        index_source = toc_source or ("toc" if index_items else ("review_required" if explicit_toc_detected else "auto"))

    with timing_collector.stage("JSON generation", "json_generation_time"):
        save_index(req.pdf_id, classified_final)

        if record:
            pending_after = int(record.get("pending_pages", 0) or 0)
            needs_review = bool(review_reason) and pending_after == 0
            update_pdf_record(
                req.pdf_id,
                status="needs_review" if needs_review else "index_ready",
                index_ready=True,
                index_source=index_source,
                queue_bucket="reindex_review" if needs_review else ("deferred" if pending_after > 0 else "library"),
                deferred_decision="pending" if pending_after > 0 else "completed",
                review_reason=review_reason if needs_review else "",
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

    update_batch_report_for_pdf(req.pdf_id)
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
        "review_reason": (record or {}).get("review_reason", ""),
        "index_export_file": str(export_path),
    }



@app.post("/api/generate-index")
async def generate_index(req: IndexRequest):
    payload = generate_index_payload(req)
    record = get_pdf_record(req.pdf_id)
    if record:
        pending_pages = int(record.get("pending_pages", 0) or 0)
        is_fully_vectorized = pending_pages == 0 and (str(record.get("retrieval_status") or "").lower() == "vectorized" or bool(record.get("chat_ready")))
        review_reason = payload.get("review_reason", "")
        needs_review = bool(review_reason) and pending_pages == 0
        update_pdf_record(
            req.pdf_id,
            status="index_ready" if pending_pages > 0 else ("needs_review" if needs_review else ("vectorized" if is_fully_vectorized else "index_ready")),
            index_ready=True,
            index_source=payload.get("index_source", record.get("index_source", "auto")),
            indexed_pages=payload.get("indexed_pages", record.get("indexed_pages", 0)),
            selected_start_page=payload.get("indexed_page_start", record.get("selected_start_page", 1)),
            selected_end_page=payload.get("indexed_page_end", record.get("selected_end_page", 1)),
            queue_bucket="deferred" if pending_pages > 0 else ("reindex_review" if needs_review else "library"),
            deferred_decision="pending" if pending_pages > 0 else "completed",
            review_reason=review_reason if needs_review else ("" if pending_pages == 0 else record.get("review_reason", "")),
        )
        updated_record = get_pdf_record(req.pdf_id) or record
        if pending_pages == 0 and not needs_review:
            audit_saved_index_record(req.pdf_id, record=updated_record)
            updated_record = get_pdf_record(req.pdf_id) or updated_record
        payload["status"] = updated_record.get("status", payload.get("status", "index_ready"))
        payload["retrieval_status"] = updated_record.get("retrieval_status", payload.get("retrieval_status", "legacy"))
        payload["pending_pages"] = updated_record.get("pending_pages", payload.get("pending_pages", 0))
        payload["chat_ready"] = bool(updated_record.get("chat_ready", payload.get("chat_ready", True)))
        payload["review_reason"] = updated_record.get("review_reason", payload.get("review_reason", ""))
    update_batch_report_for_pdf(req.pdf_id)
    return payload

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


@app.post("/api/stage1-batch/enqueue")
async def enqueue_stage1_batch(
    files: list[UploadFile] = File(...),
    start_page: int = Form(1),
    end_page: Optional[int] = Form(None),
):
    results = []
    seen_pdf_ids: set[str] = set()
    batch_run_id = create_batch_run_id()
    batch_enqueued_at = utc_now_iso()
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
                "reason": "Duplicate PDF in the same chunk. Skipping.",
                "skipped_duplicate": True,
            })
            continue
        seen_pdf_ids.add(pdf_id)
        results.append(enqueue_pdf_for_stage1(
            pdf_bytes,
            file.filename,
            start_page=start_page,
            end_page=end_page,
            batch_run_id=batch_run_id,
            batch_enqueued_at=batch_enqueued_at,
        ))

    report_path = write_batch_report(batch_run_id)
    started_runner = start_stage1_batch_runner_if_needed()
    return {
        "pdfs": results,
        "count": len(results),
        "started_runner": started_runner,
        "runner": dict(stage1_batch_runner_status),
        "batch_run_id": batch_run_id,
        "batch_report_file": str(report_path) if report_path else "",
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
        "stage1_batch_runner": dict(stage1_batch_runner_status),
        "audit_runner": dict(audit_runner_status),
        "reindex_runner": dict(reindex_review_runner_status),
    }


@app.post("/api/queues/reset")
async def reset_queue(queue_name: str = Form(...)):
    queue_value = (queue_name or "").strip().lower()
    if queue_value not in {"index", "deferred", "reindex", "stage1_batch"}:
        raise HTTPException(400, "queue_name must be index, deferred, reindex, or stage1_batch")

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
                review_reason="" if pending_pages == 0 else record.get("review_reason", ""),
            )
            reset_count += 1
            continue

        if queue_value == "deferred":
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
            continue

        if queue_value == "stage1_batch":
            should_reset = record.get("queue_bucket") == "stage1_batch" or status in {"queued_for_stage1", "indexing_running"}
            if not should_reset:
                continue
            update_pdf_record(
                record["pdf_id"],
                status="queued_for_stage1",
                retrieval_status="queued_for_stage1",
                chat_ready=False,
                queue_bucket="stage1_batch",
                deferred_decision="queue",
                last_error="",
            )
            reset_count += 1
            continue

        should_reset = record.get("queue_bucket") == "reindex_review" or status == "needs_review" or bool(record.get("review_reason"))
        if not should_reset:
            continue
        update_pdf_record(
            record["pdf_id"],
            status="vectorized" if is_fully_vectorized_record(record) else status,
            queue_bucket="library" if is_fully_vectorized_record(record) else record.get("queue_bucket", "library"),
            review_reason="",
            last_error="",
        )
        reset_count += 1

    return {
        "queue_name": queue_value,
        "reset_count": reset_count,
        "queues": build_queue_snapshot(),
    }


@app.get("/api/pdfs/{pdf_id}/timings")
async def get_pdf_timings(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    return {
        "pdf_id": pdf_id,
        "filename": record.get("filename") or f"{pdf_id}.pdf",
        "events": get_pdf_timing_history(pdf_id),
    }


@app.get("/api/batch-reports")
async def get_batch_reports(limit: int = 12):
    safe_limit = max(1, min(int(limit or 12), 50))
    return {
        "reports": list_batch_reports(safe_limit),
    }


@app.get("/api/batch-reports/{batch_run_id}")
async def get_batch_report(batch_run_id: str):
    report_path = batch_report_path(batch_run_id)
    if not report_path.exists():
        raise HTTPException(404, f"Batch report {batch_run_id} not found")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Could not read batch report: {exc}")


@app.get("/api/pdfs/{pdf_id}")
async def get_pdf_details(pdf_id: str):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")
    saved_index = get_saved_index(pdf_id)
    refreshed = refresh_saved_index_if_needed(pdf_id, record=record, saved_index=saved_index, reason="loading pdf details")
    if refreshed:
        record = get_pdf_record(pdf_id) or record
        saved_index = refreshed.get("index", saved_index)
    return {
        "pdf": record,
        "index": saved_index,
    }


@app.post("/api/pdfs/{pdf_id}/index")
async def save_pdf_index(pdf_id: str, req: IndexSaveRequest):
    record = get_pdf_record(pdf_id)
    if not record:
        raise HTTPException(404, f"PDF {pdf_id} not found")

    normalized_items = []
    total_pages = max(1, int(record.get("total_pages") or 1))

    for raw_item in req.index or []:
        page_from = max(1, min(coerce_page_number(raw_item.get("pageFrom"), 1), total_pages))
        page_to = max(page_from, min(coerce_page_number(raw_item.get("pageTo"), page_from), total_pages))
        pdf_page_from = max(1, min(coerce_page_number(raw_item.get("pdfPageFrom"), page_from), total_pages))
        pdf_page_to = max(pdf_page_from, min(coerce_page_number(raw_item.get("pdfPageTo"), page_to), total_pages))

        normalized_items.append({
            **raw_item,
            "title": str(raw_item.get("title") or "").strip(),
            "displayTitle": str(raw_item.get("displayTitle") or raw_item.get("originalTitle") or raw_item.get("title") or "").strip(),
            "originalTitle": str(raw_item.get("originalTitle") or raw_item.get("displayTitle") or raw_item.get("title") or "").strip(),
            "pageFrom": page_from,
            "pageTo": page_to,
            "pdfPageFrom": pdf_page_from,
            "pdfPageTo": pdf_page_to,
            "source": raw_item.get("source", "manual"),
            "serialNo": str(raw_item.get("serialNo", "")),
            "courtFee": str(raw_item.get("courtFee", "")),
        })

    normalized_items.sort(key=lambda item: (item.get("pageFrom", 0), item.get("pageTo", 0), item.get("title", "")))
    save_index(pdf_id, normalized_items)
    update_pdf_record(
        pdf_id,
        index_ready=True,
        index_source="manual",
        review_reason="",
        queue_bucket="library" if int(record.get("pending_pages") or 0) == 0 else record.get("queue_bucket", "library"),
        status="vectorized" if is_fully_vectorized_record(record) else record.get("status", "index_ready"),
    )

    export_path = export_index_json(
        pdf_id=pdf_id,
        record=get_pdf_record(pdf_id),
        index_items=normalized_items,
        indexed_start=int(record.get("selected_start_page") or 1),
        indexed_end=int(record.get("selected_end_page") or total_pages),
        total_pages=total_pages,
        index_source="manual",
    )

    return {
        "pdf_id": pdf_id,
        "index": normalized_items,
        "index_entries": len(normalized_items),
        "index_source": "manual",
        "export_path": str(export_path),
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
        with fitz.open(pdf_path) as probe_doc:
            _ = probe_doc.page_count
    with timing_collector.stage("full text extraction", "full_text_extraction_time"):
        pages_data, stats = extract_pages_from_pdf_parallel(
            pdf_path,
            pending_page_numbers,
            record["total_pages"],
            dpi=250,
            timing_collector=timing_collector,
            worker_count=OCR_WORKER_COUNT,
            pdf_id=pdf_id,
        )

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

    refreshed_record = get_pdf_record(pdf_id)
    refreshed_payload = refresh_saved_index_if_needed(pdf_id, record=refreshed_record, reason="deferred ingestion")
    refreshed_record = get_pdf_record(pdf_id) or refreshed_record or record

    timing_collector.log_summary("deferred_vectorization")

    return {
        "pdf_id": pdf_id,
        "status": refreshed_record.get("status", "vectorized"),
        "retrieval_status": refreshed_record.get("retrieval_status", "vectorized"),
        "pending_pages": refreshed_record.get("pending_pages", 0),
        "processed_pages": len(pages_data),
        "indexed_pages": refreshed_record.get("indexed_pages", updated_indexed_pages),
        "ocr_pages": stats["ocr_pages"],
        "vision_ocr_pages": stats["vision_ocr_pages"],
        "handwriting_suspected_pages": stats["handwriting_suspected_pages"],
        "chat_ready": bool(refreshed_record.get("chat_ready", True)),
        "index_source": (refreshed_payload or {}).get("index_source", refreshed_record.get("index_source", "auto")),
    }

def run_deferred_queue_worker():
    deferred_runner_status.update({
        "running": True,
        "processed": 0,
        "total": len(list_pending_pdf_ids()),
        "current_pdf_id": "",
        "current_filename": "",
        "last_error": "",
        "heartbeat_ts": time.time(),
        "paused": False,
    })
    completed = 0
    try:
        while True:
            if deferred_runner_status.get("pause_requested"):
                deferred_runner_status.update({
                    "paused": True,
                    "processed": completed,
                    "current_pdf_id": "",
                    "current_filename": "",
                    "heartbeat_ts": time.time(),
                })
                break

            pending_ids = list_pending_pdf_ids()
            if not pending_ids:
                break

            pdf_id = pending_ids[0]
            record = get_pdf_record(pdf_id) or {}
            deferred_runner_status.update({
                "current_pdf_id": pdf_id,
                "current_filename": record.get("filename", ""),
                "processed": completed,
                "total": completed + len(pending_ids),
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


@app.post("/api/index-audit-runner")
async def start_index_audit_runner(
    search: str = Form(""),
    batch_filter: str = Form(""),
    row_start: int = Form(1),
    row_end: Optional[int] = Form(None),
):
    with audit_runner_lock:
        if audit_runner_status.get("running"):
            return {
                "started": False,
                "runner": dict(audit_runner_status),
                "message": "Index audit is already running.",
            }
        records = filter_pdf_records_for_audit(search=search, batch_filter=batch_filter, row_start=row_start, row_end=row_end)
        if not records:
            return {
                "started": False,
                "runner": dict(audit_runner_status),
                "message": "No fully vectorized PDFs matched the audit filters.",
            }
        worker = Thread(target=run_index_audit_worker, args=(records,), daemon=True)
        worker.start()
    return {
        "started": True,
        "runner": dict(audit_runner_status),
        "count": len(records),
    }


@app.post("/api/reindex-review-runner")
async def start_reindex_review_runner():
    with reindex_review_runner_lock:
        if reindex_review_runner_status.get("running"):
            return {
                "started": False,
                "runner": dict(reindex_review_runner_status),
                "message": "Reindex review queue is already running.",
            }
        pdf_ids = list_reindex_review_pdf_ids()
        if not pdf_ids:
            return {
                "started": False,
                "runner": dict(reindex_review_runner_status),
                "message": "No PDFs are waiting in the reindex review queue.",
            }
        worker = Thread(target=run_reindex_review_worker, args=(pdf_ids,), daemon=True)
        worker.start()
    return {
        "started": True,
        "runner": dict(reindex_review_runner_status),
        "count": len(pdf_ids),
    }


@app.post("/api/stage1-batch-runner")
async def start_stage1_batch_background():
    started = start_stage1_batch_runner_if_needed()
    if not started:
        return {
            "started": False,
            "runner": dict(stage1_batch_runner_status),
            "message": "No PDFs are waiting in the batch indexing queue." if not list_stage1_batch_pdf_ids() else "Batch indexing queue is already running.",
        }
    return {
        "started": True,
        "runner": dict(stage1_batch_runner_status),
        "count": len(list_stage1_batch_pdf_ids()),
    }


def start_deferred_runner_if_needed(force_resume: bool = False) -> bool:
    with deferred_runner_lock:
        if deferred_runner_status.get("running"):
            return False
        if deferred_runner_status.get("paused") and not force_resume:
            return False
        pdf_ids = list_pending_pdf_ids()
        if not pdf_ids:
            return False
        deferred_runner_status.update({
            "pause_requested": False,
            "paused": False,
            "stop_requested": False,
            "processed": 0,
            "total": len(pdf_ids),
            "heartbeat_ts": time.time(),
        })
        worker = Thread(target=run_deferred_queue_worker, daemon=True)
        worker.start()
        return True


@app.post("/api/process-pending-runner")
async def process_pending_background():
    started = start_deferred_runner_if_needed(force_resume=True)
    if not started:
        return {
            "started": False,
            "runner": dict(deferred_runner_status),
            "message": "No PDFs are waiting in the deferred queue." if not list_pending_pdf_ids() else "Deferred queue is already running.",
        }
    return {
        "started": True,
        "runner": dict(deferred_runner_status),
        "count": len(list_pending_pdf_ids()),
    }


@app.post("/api/runners/control-all")
async def control_all_runners(action: str = Form(...)):
    action_value = (action or "").strip().lower()
    if action_value != "stop":
        raise HTTPException(400, "action must be stop")

    messages = []

    with deferred_runner_lock:
        if deferred_runner_status.get("running"):
            deferred_runner_status.update({
                "pause_requested": True,
                "stop_requested": True,
                "heartbeat_ts": time.time(),
            })
            messages.append("Deferred queue will stop after the current PDF.")

    with stage1_batch_runner_lock:
        if stage1_batch_runner_status.get("running"):
            stage1_batch_runner_status.update({
                "stop_requested": True,
                "status": "stopping",
                "heartbeat_ts": time.time(),
            })
            messages.append("Batch indexing queue will stop after the current PDF.")

    with audit_runner_lock:
        if audit_runner_status.get("running"):
            audit_runner_status.update({
                "stop_requested": True,
                "status": "stopping",
                "heartbeat_ts": time.time(),
            })
            messages.append("Audit runner will stop after the current PDF.")

    with reindex_review_runner_lock:
        if reindex_review_runner_status.get("running"):
            reindex_review_runner_status.update({
                "stop_requested": True,
                "status": "stopping",
                "heartbeat_ts": time.time(),
            })
            messages.append("Review reindex runner will stop after the current PDF.")

    with index_runner_lock:
        if index_runner_status.get("running"):
            index_runner_status.update({
                "stop_requested": True,
                "status": "stopping",
                "heartbeat_ts": time.time(),
            })
            messages.append("Current indexing job cannot be killed mid-file safely; it will finish the current PDF only.")

    return {
        "accepted": bool(messages),
        "message": " ".join(messages) if messages else "No background runner is active.",
        "runner": dict(deferred_runner_status),
        "index_runner": dict(index_runner_status),
        "stage1_batch_runner": dict(stage1_batch_runner_status),
        "audit_runner": dict(audit_runner_status),
        "reindex_runner": dict(reindex_review_runner_status),
    }


@app.post("/api/process-pending-runner/control")
async def control_process_pending_runner(action: str = Form(...)):
    action_value = (action or "").strip().lower()
    if action_value not in {"stop", "resume"}:
        raise HTTPException(400, "action must be stop or resume")

    if action_value == "stop":
        with deferred_runner_lock:
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

    with deferred_runner_lock:
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

    started = start_deferred_runner_if_needed(force_resume=True)
    return {
        "accepted": bool(started),
        "message": "Deferred queue resumed." if started else "Deferred queue is already running.",
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
    normalize_background_queue_state()
    resumed_stage1 = start_stage1_batch_runner_if_needed()
    resumed_deferred = start_deferred_runner_if_needed(force_resume=True)
    if not LOCAL_TEXT_MODEL:
        log.warning("LOCAL_TEXT_MODEL is not set - AI features will fail")
    log.info(f"Local LLM URL: {LOCAL_LLM_BASE_URL}")
    log.info(f"Vision model : {LOCAL_VISION_MODEL}")
    log.info(f"Text model   : {LOCAL_TEXT_MODEL}")
    log.info(f"ChromaDB path: {CHROMA_DB_PATH}")
    log.info(
        "Embedding config: model=%s preferred_device=%s embedding_batch_size=%s db_batch_size=%s ocr_worker_count=%s",
        EMBEDDING_MODEL_NAME,
        resolve_embedding_device(),
        EMBEDDING_BATCH_SIZE,
        VECTOR_DB_BATCH_SIZE,
        OCR_WORKER_COUNT,
    )
    await asyncio.to_thread(get_embedder)
    if resumed_stage1:
        log.info("Resumed batch indexing queue on startup")
    if resumed_deferred:
        log.info("Resumed deferred queue on startup")
    log.info("Server ready")
