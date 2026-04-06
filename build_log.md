# VerdictAI — Build Log

---

## Session 0 — 2026-04-05

**Goal:** Repository scaffolding and pre-build setup.

**Completed:**
- Created folder structure: `backend/`, `frontend/`, `backend/eval/`, `backend/judge/`, `backend/db/`
- Created `.gitignore` (Python, .env, SQLite, IDE exclusions)
- Created `.env.example` with `DEV_MODE`, API key placeholders, and `DATABASE_URL`
- Created `backend/main.py` — FastAPI stub with `/health` endpoint
- Created `frontend/app.py` — Streamlit stub with title and "Coming soon" placeholder
- Created `backend/requirements.txt` — all backend dependencies
- Created `frontend/requirements.txt` — all frontend dependencies
- Preserved existing `pricing.yaml` and `models.yaml` (already populated)
- Updated `README.md`

**Notes:**
- `DEV_MODE=true` in `.env` bypasses real API calls and uses mock scores from `models.yaml`
- `pricing.yaml` and `models.yaml` are fully specified — no changes needed until Session 1
- Stack: FastAPI (backend) + Streamlit (frontend) + SQLite (db)
- Deployment targets: Render (backend) + Streamlit Community Cloud (frontend)

**Next session:** Session 1 — Provider abstraction layer + database schema
