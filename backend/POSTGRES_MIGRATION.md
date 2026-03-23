# Postgres Migration Guide

## What Moves
- `pdf_records`
- `extracted_pages`
- `saved_indexes`

## What Stays As-Is
- `backend/chroma_db` (existing vectors)
- `backend/stored_pdfs`
- `backend/index_exports`

## 1. Back Up Current Data
- `backend/workflow.db`
- `backend/chroma_db/`
- `backend/stored_pdfs/`
- `backend/index_exports/`

## 2. Install Dependency
```bash
pip install "psycopg[binary]"
```

## 3. Create Postgres Database
```sql
CREATE DATABASE court_rag;
```

## 4. Set Environment
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/court_rag
WORKFLOW_SQLITE_PATH=./workflow.db
```

## 5. Run the Migration
```bash
cd backend
python migrate_sqlite_to_postgres.py
```

## 6. Start the Backend on Postgres
Once `DATABASE_URL` is set, `workflow_state.py` automatically uses Postgres.

```bash
python -m uvicorn main:app --reload --port 8000
```

## 7. Verify
- `/health`
- `/api/pdfs`
- `/api/queues`
- open an old indexed PDF
- chat against an already-vectorized PDF

## Rollback
Remove `DATABASE_URL` from `backend/.env` and restart the backend. It will fall back to SQLite.
