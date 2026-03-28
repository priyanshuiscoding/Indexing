from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
SQLITE_DB_PATH = Path(os.getenv("WORKFLOW_SQLITE_PATH") or (Path(__file__).resolve().parent / "workflow.db"))
DB_PATH = SQLITE_DB_PATH
STORAGE_BACKEND = "postgres" if DATABASE_URL else "sqlite"
STORAGE_TARGET = DATABASE_URL or str(SQLITE_DB_PATH)
POSTGRES_SCHEMA_PATH = Path(__file__).resolve().with_name("postgres_schema.sql")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_postgres() -> bool:
    return STORAGE_BACKEND == "postgres"


def get_connection():
    if _is_postgres():
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(DATABASE_URL, row_factory=dict_row)

    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, ddl: str):
    if column_name not in _column_names(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def _init_sqlite_db():
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS pdf_records (
                pdf_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                total_pages INTEGER NOT NULL,
                selected_start_page INTEGER NOT NULL,
                selected_end_page INTEGER NOT NULL,
                indexed_pages INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                retrieval_status TEXT NOT NULL,
                index_ready INTEGER NOT NULL DEFAULT 0,
                chat_ready INTEGER NOT NULL DEFAULT 0,
                pending_pages INTEGER NOT NULL DEFAULT 0,
                index_source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS extracted_pages (
                pdf_id TEXT NOT NULL,
                page_num INTEGER NOT NULL,
                text TEXT NOT NULL,
                used_ocr INTEGER NOT NULL DEFAULT 0,
                vision_used INTEGER NOT NULL DEFAULT 0,
                handwriting_suspected INTEGER NOT NULL DEFAULT 0,
                extraction_method TEXT NOT NULL,
                stage TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (pdf_id, page_num)
            );

            CREATE TABLE IF NOT EXISTS saved_indexes (
                pdf_id TEXT PRIMARY KEY,
                index_json TEXT NOT NULL,
                total_entries INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "pdf_records", "cnr_number", "cnr_number TEXT DEFAULT ''")
        _ensure_column(conn, "pdf_records", "file_size_bytes", "file_size_bytes INTEGER DEFAULT 0")
        _ensure_column(conn, "pdf_records", "queue_bucket", "queue_bucket TEXT DEFAULT 'index'")
        _ensure_column(conn, "pdf_records", "deferred_decision", "deferred_decision TEXT DEFAULT 'pending'")
        _ensure_column(conn, "pdf_records", "last_error", "last_error TEXT DEFAULT ''")
        _ensure_column(conn, "pdf_records", "review_reason", "review_reason TEXT DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def _init_postgres_db():
    schema_sql = POSTGRES_SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
            cur.execute("ALTER TABLE pdf_records ADD COLUMN IF NOT EXISTS review_reason TEXT NOT NULL DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def init_db():
    if _is_postgres():
        _init_postgres_db()
    else:
        _init_sqlite_db()


def _row_to_record(row) -> Optional[dict]:
    if row is None:
        return None
    record = dict(row)
    record["index_ready"] = bool(record.get("index_ready"))
    record["chat_ready"] = bool(record.get("chat_ready"))
    return record


def get_pdf_record(pdf_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM pdf_records WHERE pdf_id = %s", (pdf_id,))
                row = cur.fetchone()
        else:
            row = conn.execute("SELECT * FROM pdf_records WHERE pdf_id = ?", (pdf_id,)).fetchone()
        return _row_to_record(row)
    finally:
        conn.close()


def upsert_pdf_record(
    pdf_id: str,
    filename: str,
    total_pages: int,
    selected_start_page: int,
    selected_end_page: int,
    indexed_pages: int,
    status: str,
    retrieval_status: str,
    index_ready: bool = False,
    chat_ready: bool = False,
    pending_pages: Optional[int] = None,
    index_source: str = "",
    cnr_number: str = "",
    file_size_bytes: int = 0,
    queue_bucket: str = "index",
    deferred_decision: str = "pending",
    last_error: str = "",
    review_reason: str = "",
):
    now = utc_now_iso()
    pending_pages = pending_pages if pending_pages is not None else max(total_pages - indexed_pages, 0)
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pdf_records (
                        pdf_id, filename, cnr_number, file_size_bytes, total_pages, selected_start_page, selected_end_page,
                        indexed_pages, status, retrieval_status, index_ready, chat_ready,
                        pending_pages, index_source, queue_bucket, deferred_decision, last_error, review_reason, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pdf_id) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        cnr_number = EXCLUDED.cnr_number,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        total_pages = EXCLUDED.total_pages,
                        selected_start_page = EXCLUDED.selected_start_page,
                        selected_end_page = EXCLUDED.selected_end_page,
                        indexed_pages = EXCLUDED.indexed_pages,
                        status = EXCLUDED.status,
                        retrieval_status = EXCLUDED.retrieval_status,
                        index_ready = EXCLUDED.index_ready,
                        chat_ready = EXCLUDED.chat_ready,
                        pending_pages = EXCLUDED.pending_pages,
                        index_source = EXCLUDED.index_source,
                        queue_bucket = EXCLUDED.queue_bucket,
                        deferred_decision = EXCLUDED.deferred_decision,
                        last_error = EXCLUDED.last_error,
                        review_reason = EXCLUDED.review_reason,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        pdf_id, filename, cnr_number, file_size_bytes, total_pages, selected_start_page,
                        selected_end_page, indexed_pages, status, retrieval_status, bool(index_ready), bool(chat_ready),
                        pending_pages, index_source, queue_bucket, deferred_decision, last_error, review_reason, now, now,
                    ),
                )
        else:
            conn.execute(
                """
                INSERT INTO pdf_records (
                    pdf_id, filename, cnr_number, file_size_bytes, total_pages, selected_start_page, selected_end_page,
                    indexed_pages, status, retrieval_status, index_ready, chat_ready,
                    pending_pages, index_source, queue_bucket, deferred_decision, last_error, review_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id) DO UPDATE SET
                    filename = excluded.filename,
                    cnr_number = excluded.cnr_number,
                    file_size_bytes = excluded.file_size_bytes,
                    total_pages = excluded.total_pages,
                    selected_start_page = excluded.selected_start_page,
                    selected_end_page = excluded.selected_end_page,
                    indexed_pages = excluded.indexed_pages,
                    status = excluded.status,
                    retrieval_status = excluded.retrieval_status,
                    index_ready = excluded.index_ready,
                    chat_ready = excluded.chat_ready,
                    pending_pages = excluded.pending_pages,
                    index_source = excluded.index_source,
                    queue_bucket = excluded.queue_bucket,
                    deferred_decision = excluded.deferred_decision,
                    last_error = excluded.last_error,
                    review_reason = excluded.review_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    pdf_id, filename, cnr_number, file_size_bytes, total_pages, selected_start_page,
                    selected_end_page, indexed_pages, status, retrieval_status, int(index_ready), int(chat_ready),
                    pending_pages, index_source, queue_bucket, deferred_decision, last_error, review_reason, now, now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def update_pdf_record(pdf_id: str, **fields):
    if not fields:
        return
    fields["updated_at"] = utc_now_iso()
    conn = get_connection()
    try:
        if _is_postgres():
            assignments = ", ".join(f"{key} = %s" for key in fields)
            values = [bool(value) if isinstance(value, bool) else value for value in fields.values()]
            with conn.cursor() as cur:
                cur.execute(f"UPDATE pdf_records SET {assignments} WHERE pdf_id = %s", values + [pdf_id])
        else:
            assignments = ", ".join(f"{key} = ?" for key in fields)
            values = [int(value) if isinstance(value, bool) else value for value in fields.values()]
            conn.execute(f"UPDATE pdf_records SET {assignments} WHERE pdf_id = ?", values + [pdf_id])
        conn.commit()
    finally:
        conn.close()

def replace_extracted_pages(pdf_id: str, pages_data: list[dict], stage: str):
    now = utc_now_iso()
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM extracted_pages WHERE pdf_id = %s", (pdf_id,))
                cur.executemany(
                    """
                    INSERT INTO extracted_pages (
                        pdf_id, page_num, text, used_ocr, vision_used,
                        handwriting_suspected, extraction_method, stage, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            pdf_id, page["page_num"], page["text"], bool(page["used_ocr"]), bool(page["vision_used"]),
                            bool(page["handwriting_suspected"]), page["extraction_method"], stage, now, now,
                        )
                        for page in pages_data
                    ],
                )
        else:
            conn.execute("DELETE FROM extracted_pages WHERE pdf_id = ?", (pdf_id,))
            conn.executemany(
                """
                INSERT INTO extracted_pages (
                    pdf_id, page_num, text, used_ocr, vision_used,
                    handwriting_suspected, extraction_method, stage, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        pdf_id, page["page_num"], page["text"], int(page["used_ocr"]), int(page["vision_used"]),
                        int(page["handwriting_suspected"]), page["extraction_method"], stage, now, now,
                    )
                    for page in pages_data
                ],
            )
        conn.commit()
    finally:
        conn.close()


def upsert_extracted_pages(pdf_id: str, pages_data: list[dict], stage: str):
    now = utc_now_iso()
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO extracted_pages (
                        pdf_id, page_num, text, used_ocr, vision_used,
                        handwriting_suspected, extraction_method, stage, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pdf_id, page_num) DO UPDATE SET
                        text = EXCLUDED.text,
                        used_ocr = EXCLUDED.used_ocr,
                        vision_used = EXCLUDED.vision_used,
                        handwriting_suspected = EXCLUDED.handwriting_suspected,
                        extraction_method = EXCLUDED.extraction_method,
                        stage = EXCLUDED.stage,
                        updated_at = EXCLUDED.updated_at
                    """,
                    [
                        (
                            pdf_id, page["page_num"], page["text"], bool(page["used_ocr"]), bool(page["vision_used"]),
                            bool(page["handwriting_suspected"]), page["extraction_method"], stage, now, now,
                        )
                        for page in pages_data
                    ],
                )
        else:
            conn.executemany(
                """
                INSERT INTO extracted_pages (
                    pdf_id, page_num, text, used_ocr, vision_used,
                    handwriting_suspected, extraction_method, stage, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id, page_num) DO UPDATE SET
                    text = excluded.text,
                    used_ocr = excluded.used_ocr,
                    vision_used = excluded.vision_used,
                    handwriting_suspected = excluded.handwriting_suspected,
                    extraction_method = excluded.extraction_method,
                    stage = excluded.stage,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        pdf_id, page["page_num"], page["text"], int(page["used_ocr"]), int(page["vision_used"]),
                        int(page["handwriting_suspected"]), page["extraction_method"], stage, now, now,
                    )
                    for page in pages_data
                ],
            )
        conn.commit()
    finally:
        conn.close()


def get_cached_pages(pdf_id: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> list[dict]:
    conn = get_connection()
    try:
        if _is_postgres():
            clauses = ["pdf_id = %s"]
            params = [pdf_id]
            if start_page is not None:
                clauses.append("page_num >= %s")
                params.append(start_page)
            if end_page is not None:
                clauses.append("page_num <= %s")
                params.append(end_page)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT page_num, text, used_ocr, vision_used, handwriting_suspected, extraction_method, stage
                    FROM extracted_pages
                    WHERE {' AND '.join(clauses)}
                    ORDER BY page_num
                    """,
                    params,
                )
                rows = cur.fetchall()
        else:
            clauses = ["pdf_id = ?"]
            params = [pdf_id]
            if start_page is not None:
                clauses.append("page_num >= ?")
                params.append(start_page)
            if end_page is not None:
                clauses.append("page_num <= ?")
                params.append(end_page)
            rows = conn.execute(
                f"""
                SELECT page_num, text, used_ocr, vision_used, handwriting_suspected, extraction_method, stage
                FROM extracted_pages
                WHERE {' AND '.join(clauses)}
                ORDER BY page_num
                """,
                params,
            ).fetchall()
        return [
            {
                "page_num": row["page_num"],
                "text": row["text"],
                "used_ocr": bool(row["used_ocr"]),
                "vision_used": bool(row["vision_used"]),
                "handwriting_suspected": bool(row["handwriting_suspected"]),
                "extraction_method": row["extraction_method"],
                "stage": row["stage"],
            }
            for row in rows
        ]
    finally:
        conn.close()


def save_index(pdf_id: str, index_items: list[dict]):
    now = utc_now_iso()
    payload = json.dumps(index_items or [], ensure_ascii=False)
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO saved_indexes (pdf_id, index_json, total_entries, created_at, updated_at)
                    VALUES (%s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (pdf_id) DO UPDATE SET
                        index_json = EXCLUDED.index_json,
                        total_entries = EXCLUDED.total_entries,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (pdf_id, payload, len(index_items or []), now, now),
                )
        else:
            conn.execute(
                """
                INSERT INTO saved_indexes (pdf_id, index_json, total_entries, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id) DO UPDATE SET
                    index_json = excluded.index_json,
                    total_entries = excluded.total_entries,
                    updated_at = excluded.updated_at
                """,
                (pdf_id, payload, len(index_items or []), now, now),
            )
        conn.commit()
    finally:
        conn.close()


def get_saved_index(pdf_id: str) -> list[dict]:
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("SELECT index_json FROM saved_indexes WHERE pdf_id = %s", (pdf_id,))
                row = cur.fetchone()
        else:
            row = conn.execute("SELECT index_json FROM saved_indexes WHERE pdf_id = ?", (pdf_id,)).fetchone()
        if not row:
            return []
        payload = row["index_json"]
        if isinstance(payload, str):
            return json.loads(payload or "[]")
        return payload or []
    finally:
        conn.close()

def list_pdf_records(search: str = "") -> list[dict]:
    conn = get_connection()
    try:
        if _is_postgres():
            clauses = ["1 = 1"]
            params: list[object] = []
            if search.strip():
                like = f"%{search.strip()}%"
                clauses.append("(filename ILIKE %s OR cnr_number ILIKE %s OR pdf_id ILIKE %s)")
                params.extend([like, like, like])
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT pdf_id, filename, cnr_number, file_size_bytes, total_pages, indexed_pages,
                           selected_start_page, selected_end_page, status, retrieval_status,
                           index_ready, chat_ready, pending_pages, index_source, queue_bucket,
                           deferred_decision, last_error, review_reason, updated_at
                    FROM pdf_records
                    WHERE {' AND '.join(clauses)}
                    ORDER BY updated_at DESC
                    """,
                    params,
                )
                rows = cur.fetchall()
        else:
            clauses = ["1 = 1"]
            params: list[object] = []
            if search.strip():
                like = f"%{search.strip()}%"
                clauses.append("(filename LIKE ? OR cnr_number LIKE ? OR pdf_id LIKE ?)")
                params.extend([like, like, like])
            rows = conn.execute(
                f"""
                SELECT pdf_id, filename, cnr_number, file_size_bytes, total_pages, indexed_pages,
                       selected_start_page, selected_end_page, status, retrieval_status,
                       index_ready, chat_ready, pending_pages, index_source, queue_bucket,
                       deferred_decision, last_error, review_reason, updated_at
                FROM pdf_records
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                """,
                params,
            ).fetchall()
        return [_row_to_record(row) for row in rows]
    finally:
        conn.close()


def list_pending_pdf_ids() -> list[str]:
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pdf_id
                    FROM pdf_records
                    WHERE pending_pages > 0
                      AND deferred_decision != %s
                      AND queue_bucket != %s
                    ORDER BY updated_at ASC
                    """,
                    ("skip", "stage1_batch"),
                )
                rows = cur.fetchall()
        else:
            rows = conn.execute(
                """
                SELECT pdf_id
                FROM pdf_records
                WHERE pending_pages > 0
                  AND deferred_decision != 'skip'
                  AND queue_bucket != 'stage1_batch'
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return [row["pdf_id"] for row in rows]
    finally:
        conn.close()


def list_reindex_review_pdf_ids() -> list[str]:
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pdf_id FROM pdf_records WHERE queue_bucket = %s ORDER BY updated_at ASC",
                    ("reindex_review",),
                )
                rows = cur.fetchall()
        else:
            rows = conn.execute(
                "SELECT pdf_id FROM pdf_records WHERE queue_bucket = 'reindex_review' ORDER BY updated_at ASC"
            ).fetchall()
        return [row["pdf_id"] for row in rows]
    finally:
        conn.close()


def list_stage1_batch_pdf_ids() -> list[str]:
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pdf_id
                    FROM pdf_records
                    WHERE queue_bucket = %s
                      AND status IN (%s, %s)
                    ORDER BY updated_at ASC
                    """,
                    ("stage1_batch", "queued_for_stage1", "indexing_running"),
                )
                rows = cur.fetchall()
        else:
            rows = conn.execute(
                """
                SELECT pdf_id
                FROM pdf_records
                WHERE queue_bucket = 'stage1_batch'
                  AND status IN ('queued_for_stage1', 'indexing_running')
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return [row["pdf_id"] for row in rows]
    finally:
        conn.close()


def build_queue_snapshot() -> dict:
    records = list_pdf_records()
    return {
        "index_ready": [record for record in records if record.get("index_ready")],
        "stage1_batch": [record for record in records if record.get("queue_bucket") == "stage1_batch"],
        "pending_vectorization": [record for record in records if record.get("pending_pages", 0) > 0 and record.get("queue_bucket") != "stage1_batch"],
        "vectorized": [record for record in records if record.get("chat_ready")],
        "reindex_review": [record for record in records if record.get("queue_bucket") == "reindex_review"],
        "errors": [record for record in records if record.get("last_error")],
    }


def delete_pdf_state(pdf_id: str):
    conn = get_connection()
    try:
        if _is_postgres():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM extracted_pages WHERE pdf_id = %s", (pdf_id,))
                cur.execute("DELETE FROM saved_indexes WHERE pdf_id = %s", (pdf_id,))
                cur.execute("DELETE FROM pdf_records WHERE pdf_id = %s", (pdf_id,))
        else:
            conn.execute("DELETE FROM extracted_pages WHERE pdf_id = ?", (pdf_id,))
            conn.execute("DELETE FROM saved_indexes WHERE pdf_id = ?", (pdf_id,))
            conn.execute("DELETE FROM pdf_records WHERE pdf_id = ?", (pdf_id,))
        conn.commit()
    finally:
        conn.close()
