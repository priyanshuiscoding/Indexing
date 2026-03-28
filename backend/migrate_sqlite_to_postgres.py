from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
SQLITE_DB_PATH = Path(os.getenv("WORKFLOW_SQLITE_PATH") or (ROOT / "workflow.db"))
DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
SCHEMA_PATH = ROOT / "postgres_schema.sql"


def require_config():
    if not SQLITE_DB_PATH.exists():
        raise SystemExit(f"SQLite workflow DB not found: {SQLITE_DB_PATH}")
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is required to migrate into Postgres")


def sqlite_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def postgres_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_postgres(conn):
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def migrate_pdf_records(src, dst):
    rows = src.execute("SELECT * FROM pdf_records ORDER BY updated_at ASC").fetchall()
    with dst.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO pdf_records (
                pdf_id, filename, cnr_number, file_size_bytes, total_pages, selected_start_page, selected_end_page,
                indexed_pages, status, retrieval_status, index_ready, chat_ready,
                pending_pages, index_source, queue_bucket, deferred_decision, last_error, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                updated_at = EXCLUDED.updated_at
            """,
            [
                (
                    row["pdf_id"], row["filename"], row["cnr_number"] if "cnr_number" in row.keys() else "",
                    row["file_size_bytes"] if "file_size_bytes" in row.keys() else 0, row["total_pages"],
                    row["selected_start_page"], row["selected_end_page"], row["indexed_pages"], row["status"],
                    row["retrieval_status"], bool(row["index_ready"]), bool(row["chat_ready"]), row["pending_pages"],
                    row["index_source"] or "", row["queue_bucket"] if "queue_bucket" in row.keys() else "index",
                    row["deferred_decision"] if "deferred_decision" in row.keys() else "pending",
                    row["last_error"] if "last_error" in row.keys() else "", row["created_at"], row["updated_at"],
                )
                for row in rows
            ],
        )
    dst.commit()
    return len(rows)


def migrate_extracted_pages(src, dst):
    rows = src.execute("SELECT * FROM extracted_pages ORDER BY pdf_id ASC, page_num ASC").fetchall()
    with dst.cursor() as cur:
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
                    row["pdf_id"], row["page_num"], row["text"], bool(row["used_ocr"]), bool(row["vision_used"]),
                    bool(row["handwriting_suspected"]), row["extraction_method"], row["stage"], row["created_at"], row["updated_at"],
                )
                for row in rows
            ],
        )
    dst.commit()
    return len(rows)

def migrate_saved_indexes(src, dst):
    rows = src.execute("SELECT * FROM saved_indexes ORDER BY updated_at ASC").fetchall()
    with dst.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO saved_indexes (pdf_id, index_json, total_entries, created_at, updated_at)
            VALUES (%s, %s::jsonb, %s, %s, %s)
            ON CONFLICT (pdf_id) DO UPDATE SET
                index_json = EXCLUDED.index_json,
                total_entries = EXCLUDED.total_entries,
                updated_at = EXCLUDED.updated_at
            """,
            [
                (
                    row["pdf_id"],
                    json.dumps(json.loads(row["index_json"] or "[]"), ensure_ascii=False),
                    row["total_entries"],
                    row["created_at"],
                    row["updated_at"],
                )
                for row in rows
            ],
        )
    dst.commit()
    return len(rows)


def count_postgres(dst, table_name: str) -> int:
    with dst.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS total FROM {table_name}")
        row = cur.fetchone()
    return int(row["total"])


def main():
    require_config()
    src = sqlite_conn()
    dst = postgres_conn()
    try:
        init_postgres(dst)
        pdf_count = migrate_pdf_records(src, dst)
        page_count = migrate_extracted_pages(src, dst)
        index_count = migrate_saved_indexes(src, dst)

        print(f"SQLite pdf_records migrated: {pdf_count}")
        print(f"SQLite extracted_pages migrated: {page_count}")
        print(f"SQLite saved_indexes migrated: {index_count}")
        print(f"Postgres pdf_records count: {count_postgres(dst, 'pdf_records')}")
        print(f"Postgres extracted_pages count: {count_postgres(dst, 'extracted_pages')}")
        print(f"Postgres saved_indexes count: {count_postgres(dst, 'saved_indexes')}")
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
