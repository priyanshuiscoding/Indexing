CREATE TABLE IF NOT EXISTS pdf_records (
    pdf_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    cnr_number TEXT NOT NULL DEFAULT '',
    file_size_bytes BIGINT NOT NULL DEFAULT 0,
    total_pages INTEGER NOT NULL,
    selected_start_page INTEGER NOT NULL,
    selected_end_page INTEGER NOT NULL,
    indexed_pages INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    retrieval_status TEXT NOT NULL,
    index_ready BOOLEAN NOT NULL DEFAULT FALSE,
    chat_ready BOOLEAN NOT NULL DEFAULT FALSE,
    pending_pages INTEGER NOT NULL DEFAULT 0,
    index_source TEXT NOT NULL DEFAULT '',
    queue_bucket TEXT NOT NULL DEFAULT 'index',
    deferred_decision TEXT NOT NULL DEFAULT 'pending',
    last_error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS extracted_pages (
    pdf_id TEXT NOT NULL REFERENCES pdf_records(pdf_id) ON DELETE CASCADE,
    page_num INTEGER NOT NULL,
    text TEXT NOT NULL,
    used_ocr BOOLEAN NOT NULL DEFAULT FALSE,
    vision_used BOOLEAN NOT NULL DEFAULT FALSE,
    handwriting_suspected BOOLEAN NOT NULL DEFAULT FALSE,
    extraction_method TEXT NOT NULL,
    stage TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (pdf_id, page_num)
);

CREATE TABLE IF NOT EXISTS saved_indexes (
    pdf_id TEXT PRIMARY KEY REFERENCES pdf_records(pdf_id) ON DELETE CASCADE,
    index_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_entries INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pdf_records_updated_at ON pdf_records (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pdf_records_queue_bucket ON pdf_records (queue_bucket);
CREATE INDEX IF NOT EXISTS idx_pdf_records_pending_pages ON pdf_records (pending_pages);
CREATE INDEX IF NOT EXISTS idx_pdf_records_filename ON pdf_records (filename);
CREATE INDEX IF NOT EXISTS idx_pdf_records_cnr_number ON pdf_records (cnr_number);
CREATE INDEX IF NOT EXISTS idx_extracted_pages_pdf_page ON extracted_pages (pdf_id, page_num);
