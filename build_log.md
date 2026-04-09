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

---

## Session 2 — 2026-04-06

**Stories:** 1.4 (Engineer Tagging), 1.5 (Modality Detection), 1.6 (Upload Validation)

**Completed:**

**Story 1.6 — Full Upload Validation**
- Enhanced `POST /upload` with comprehensive validation
- ZIP image file checking: validates manifest.json exists, all referenced images exist, file types are jpg/jpeg/png/webp only
- Specific, actionable error messages per issue (not generic)
- Warnings for missing optional fields (expected_output, engineer_name) — non-blocking
- Validation summary after passing: "✓ 10 prompts · Modality: text · Ground truth: 10/10 · Engineers: 3"
- Modality detection from file type: CSV/JSONL→"text", ZIP→"image_text", JSONL with data field→"structured_data"
- Added `ValidationWarning` schema to track non-blocking warnings

**Story 1.5 — Modality Detection & Model Filtering**
- New endpoint: `GET /models/compatible?modality={text|image_text|structured_data|video|audio}`
- Reads `modality_matrix` and `platform_suggestions` from `models.yaml` (no hardcoded model names)
- Returns: compatible_models (full details), incompatible_models (with reasons), suggestions (alternatives)
- Streamlit UI updated: calls `/models/compatible` after upload, filters model checkboxes based on modality, shows inline warnings if incompatible selected
- Claude Haiku 4.5 hidden for image_text (does not support images)
- Auto-suggestions work from models.yaml (e.g., "Switch to Claude Sonnet 4.6?" for Haiku on image data)
- Video/audio shown as "Coming in v1.1" disabled state

**Story 1.4 — Engineer Tagging & Run Label**
- Engineer names parsed from upload are saved to `prompts` table in SQLite (already working from Session 1)
- Added "Run label" optional text input field in Run tab (per-run, like custom_label)
- Run label saved with eval run to SQLite
- Engineer names and run labels are orthogonal: engineer_name is per-prompt, run label is per-evaluation batch
- Prompts grouped by engineer_name in results (preparation; full grouping UI in Sessions 4+)

**Backend Enhancements**
- `backend/eval/schemas.py`: Added `ValidationWarning`, `IncompatibleModel`, `ModelsCompatibleResponse` schemas
- `backend/eval/router.py`: Completely rewritten with Story 1.4/1.5/1.6 logic
  - `_parse_zip()` now validates image files with specific error messages
  - `_validate_modality_from_jsonl()` detects structured data
  - `_rows_to_prompts()` returns (prompts, warnings) tuple
  - `POST /upload` returns validation summary and warnings
  - `GET /models/compatible` endpoint reads modality_matrix from models.yaml

**Frontend Enhancements**
- `frontend/app.py` updated:
  - Upload tab: shows validation summary on success, expandable warnings section
  - Models tab: calls `/models/compatible` endpoint, filters checkboxes based on detected modality, shows incompatible warnings with auto-suggestions
  - Run tab: added "Run label" text input field (Story 1.4)
  - Session state: added `detected_modality` to track modality from upload

**Tests — 75/75 passing**
- `backend/tests/test_session_2.py` — 26 new tests covering:
  - Upload validation edge cases (missing columns, empty prompts, count limits)
  - ZIP image validation (missing files, unsupported types, valid extensions)
  - Modality detection (CSV→text, ZIP→image_text)
  - Model compatibility filtering (text vs image_text)
  - Suggestions for incompatible models
  - Engineer name persistence
  - Backward compatibility with Session 1

**Key Decisions Made This Session**
- ZIP manifest format: accept both list of dicts AND {prompts: [dicts]} for backward compat
- Modality detection: filename-based for ZIP/CSV/JSONL, content-based for JSONL data field
- Model filtering: done entirely in Streamlit via `/models/compatible` endpoint (no hardcoded names in any Python—all from models.yaml)
- Engineer tagging: per-prompt name is separate from per-run label (both tracked, both exported later)
- Warnings: non-blocking (render as st.warning() but let user proceed)

**Ideas for v1.1 (do not build)**
- Streaming upload validation with progress bar for large ZIPs
- Batch image format conversion (auto-convert unsupported formats)
- Engineer name autocomplete from previous runs
- Modality confidence score (how confident is detection)
- Nested folder support in ZIP (images/ folder not required, search recursively)

**Next session:** Session 3 — Persistence & Run History (Stories 1.7, 1.8)
