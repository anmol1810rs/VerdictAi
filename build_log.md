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

---

## Session 1 — 2026-04-06

**Stories:** 1.1 (Upload), 1.2 (API Keys), 1.3 (Rubric Configurator)

**Completed:**

**Foundation**
- `backend/__init__.py` — makes backend a proper Python package for clean imports
- `backend/config.py` — single source for DEV_MODE, DATABASE_URL, MOCK_LATENCY_MS, YAML loaders
- `pytest.ini` — asyncio_mode=auto configured at repo root
- `backend/db/database.py` — SQLAlchemy engine, SessionLocal, Base, create_tables(), get_table_names()
- `backend/db/models.py` — 4 ORM tables: EvalRun, Prompt, ModelResult, Verdict. API keys explicitly NOT stored anywhere.
- `backend/eval/schemas.py` — Pydantic models: PromptInput, UploadResponse, APIKeys, RubricWeights, EvalRunRequest, EvalRunResponse, EvalResultsResponse
- `backend/judge/mock_judge.py` — DEV_MODE mock judge, reads scores/reasoning/responses from models.yaml. MOCK_LATENCY_MS overridable via env var (set to 0 in tests).

**Story 1.1 — Dataset Upload**
- `POST /upload` — accepts .csv, .jsonl, .zip (image+text). Validates: min 5 prompts, max 100 prompts, required `prompt` column, parses optional `expected_output` and `engineer_name` columns. Returns modality, has_ground_truth, has_engineer_names flags.

**Story 1.2 — API Keys**
- `POST /keys/validate` — validates OpenAI key format (sk- prefix, ≥20 chars). Anthropic and Google keys optional. Keys NEVER written to DB — confirmed by column-name assertion in tests.

**Story 1.3 — Rubric Configurator**
- `POST /rubric/validate` — enforces hallucination ≥ 10 and weights sum = 100. Pydantic validators, descriptive error messages.
- 3 presets (customer_support, technical_documentation, data_labeling_qa) loaded from models.yaml — all verified to sum to 100 in tests.

**Eval pipeline (DEV_MODE scaffold)**
- `POST /eval/run` — creates EvalRun + Prompt rows, runs mock judge for each prompt×model, saves ModelResult and Verdict to SQLite, returns run_id + status=complete.
- `GET /eval/{run_id}/results` — fetches and returns stored results + verdict.
- `backend/main.py` — updated with lifespan (create_tables on startup), eval router included.

**Frontend**
- `frontend/app.py` — full Streamlit skeleton: sidebar API key inputs (session_state only, never persisted), tab layout (Upload / Rubric / Models / Run / Results), file uploader calling /upload, rubric slider UI with live weight validation, model selection panel, pre-run checklist, results display placeholder.

**Tests — 49/49 passing**
- `backend/tests/test_smoke.py` — 13 Layer 1 smoke tests
- `backend/tests/test_session_1.py` — 36 Layer 2 story acceptance tests

**Key decisions made this session:**
- eval/run is synchronous in DEV_MODE (mock is instant). Async multi-model runner deferred to Session 4 per plan.
- MOCK_LATENCY_MS set to 0 via env override in conftest.py so tests run in <1s.
- modality detection from upload is basic in Session 1; full detection+filtering is Session 2 (Story 1.5).
- `datetime.now(timezone.utc)` used instead of deprecated `datetime.utcnow()`.

**Ideas for v1.1 (do not build):**
- Streaming eval results via SSE so frontend shows live progress per prompt
- Upload template auto-detection based on column names

**Next session:** Session 2 — Modality Detection & Validation (Stories 1.4, 1.5, 1.6)
