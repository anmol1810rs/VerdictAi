# VerdictAI — Build Log

---

## Session 5 — 2026-04-13

**Goal:** Stories 2.3 (Cost Breakdown Enhancement) and 2.4 (Prompt Variance Analysis).

**Stories completed:** 2.3 ✓, 2.4 ✓

**New files:**
- `backend/tests/test_session_5.py` — 12 new acceptance tests

**DB schema changes (migration runs automatically — no manual DB reset required):**
- `ModelResult.variance_score` (Float, nullable) — max-min weighted score across models per prompt
- `database.py._migrate_add_columns()` — safe ALTER TABLE ADD COLUMN helper; called from `create_tables()` on startup; catches "column already exists" error silently

**Config changes:**
- `pricing.yaml` — added `meta.last_updated: "07 April 2026"` as a proper YAML key (not just comment) so frontend can display it programmatically

**Backend additions (`backend/verdict/verdict.py`):**
- `generate_cost_comparison_callout()` — auto-generates one-sentence comparison using the worth-it rule: score_delta > 1.0 AND cost_delta < $0.10 → "worth it"; score_delta < 0.5 OR cost_delta > $0.20 → "not worth it"; otherwise neutral
- `_build_cost_comparison()` — now includes `callout` key in the returned dict (stored in Verdict.cost_comparison)
- `calculate_prompt_variance()` — max(weighted_score) - min(weighted_score) across models for one prompt
- `generate_variance_insight()` — identifies dimension with largest per-prompt delta; returns one-sentence insight
- `rank_prompts_by_variance()` — returns (prompt_id, variance) sorted descending
- `get_high_variance_prompt_ids()` — returns top N prompt_ids by variance
- `save_variance_scores()` — persists variance_score to all ModelResult rows for each prompt; called after generate_verdict() in both mock and real eval paths
- `DIMENSION_INSIGHTS` — per-dimension natural-language phrases used in variance insights

**Schema changes (`backend/eval/schemas.py`):**
- `ModelResultOut` — added `prompt_text`, `tokens_in`, `tokens_out`, `variance_score` fields (all Optional for backward compat)

**Router changes (`backend/eval/router.py`):**
- `get_eval_results` — joins Prompt table to populate `prompt_text` per result row; includes `tokens_in`, `tokens_out`, `variance_score` in API response
- Both `_run_mock_eval` and `_run_real_eval_async` now call `save_variance_scores()` after verdict generation

**Frontend changes (`frontend/app.py`):**
- Cost breakdown: "Prices last updated: 07 April 2026" caption below cost table (reads from `pricing_cfg.meta.last_updated`)
- Cost breakdown: cost comparison callout shown as `st.info()` below the table (from `verdict.cost_comparison.callout`)
- Per-prompt: prompts now sorted by `variance_score` from API (falls back to client-side calculation if null)
- Per-prompt: ⚡ badge expander now shows "Models disagreed significantly on this prompt" + auto-insight sentence identifying the highest-delta dimension
- Per-prompt: `tokens_in`/`tokens_out` read from direct result fields with fallback to `tokens_used` dict

**Bug fixed (pre-existing, test_session_4.py):**
- `test_image_prompt_uses_correct_format` — was mocking `client.chat.completions.create` but runner uses `client.responses.create` (Responses API). Updated to use AsyncMock on `client.responses.create`, check `input` kwarg (not `messages`), and assert `input_image` content type (not `image_url`)

**Architecture decisions:**
- Callout stored in `cost_comparison["callout"]` (string key alongside per-model dicts) — avoids schema change to Verdict table
- Variance insight is regenerated in the frontend from results data (no new DB column needed for insight text)
- Migration on startup (`_migrate_add_columns`) chosen over Alembic — proportionate for a single-column SQLite add in an MVP
- Worth-it thresholds ($0.10 / $0.20 / 1.0 / 0.5) are named constants in verdict.py — easy to adjust per PRD rules

**Test results:** 12/12 Session 5 tests passing. 120/120 total (no regressions).

---

## Session 4 — 2026-04-12

**Goal:** Stories 2.1 (Verdict Generation) and 2.2 (LLM-as-Judge Scoring). The hardest session.

**Parts completed:** Multi-model runner, LLM-as-Judge, Verdict generation, Results UI.

**New files:**
- `backend/runner/runner.py` — async parallel runner using asyncio.gather()
- `backend/judge/judge.py` — LLM-as-Judge with retry logic, JSON parsing, hallucination flagging
- `backend/verdict/verdict.py` — weighted scoring, cost efficiency normalization, hallucination penalty, verdict text

**DB schema changes (delete verdictai.db to apply):**
- `EvalRun.progress_pct` (Float) — updated during runner (0→50→90→100%)
- `Prompt.image_data` (Text) — base64 data URI for image prompts
- `ModelResult.tokens_in` / `tokens_out` (Integer) — separate token count columns
- `Verdict.created_at` (DateTime)

**Hallucination semantics change (breaking, intentional):**
- Old (Sessions 1-3): `score >= 7` → flagged. Low score = good.
- New (Session 4): `score <= 3` → flagged. High score = good (10 = no hallucination).
- Updated `models.yaml` mock_scores.hallucination from 2.8 → 8.5.
- Updated `test_session_1.py` to remove cost_efficiency from dimension_scores check.

**Judge prompt version that worked (DEV_MODE=false):**
```
System: "You are an impartial AI evaluation judge. Your job is to score AI model responses
against a user-defined rubric. You must always return valid JSON. You must never return
scores without reasoning. You must quote specific text from the response that influenced
each score."

User: [prompt] → [response] → [rubric dimensions with weights from models.yaml] →
[ground truth if provided] → structured JSON return with scores/reasoning/evidence per dim.

Temperature: 0, max_tokens: 1000. Retry once on invalid JSON. Null scores on double failure.
```

**Architecture decisions:**
- `asyncio.run()` called from sync BackgroundTask thread — safe because FastAPI runs sync tasks in threadpool (no existing event loop in that thread)
- Provider imports (`AsyncOpenAI`, `AsyncAnthropic`) moved to module-level in runner.py to allow patch-based testing
- `cost_efficiency` removed from `ModelResult.dimension_scores` — it's a per-model derived metric calculated in verdict.py from `total_cost / weighted_quality_score`, not a per-result judge score
- Real verdict generation runs even in DEV_MODE (uses mock scores but full calculation logic) — ensures verdict.py logic is exercised in all test runs
- Hallucination disqualification: >30% of prompts flagged → model cannot win (even if highest score)

**Test results:** 20/20 Session 4 tests passing. 108/108 total (no regressions).

**Stories passing:** 2.1 ✓, 2.2 ✓

**Next session:** Session 5 — TBD (export, reporting, or additional eval features)

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

---

## Session 3 — 2026-04-09

**Stories:** 1.7 (Eval Run Persistence), 1.8 (Run History Sidebar)

**Completed:**

**Story 1.7 — Eval Run Persistence**
- `POST /eval/run` now returns immediately with `status=pending` — run row is committed to SQLite before any model call happens
- Eval execution moved to `_execute_eval()` background function, invoked via FastAPI `BackgroundTasks`
- Status flow: pending → running → complete (or failed on exception)
- New `GET /eval/{run_id}/status` endpoint — returns `{run_id, status, error_message, completed_at}`
- `error_message` (String) and `completed_at` (DateTime) added to `EvalRun` ORM model
- `engineer_names` (JSON list) added to `EvalRun` — populated from unique per-prompt engineer names at run creation
- Failed runs: background task catches all exceptions, saves `error_message[:500]` and `status=failed`
- Page refresh: re-fetching `/eval/{id}/status` always returns current DB state — no client-side reset
- Frontend: Run tab polls `/eval/{id}/status` every 2s via `time.sleep(2) + st.rerun()` while status is pending/running
- Run label: `custom_label` from Story 1.4 wired correctly to eval_runs table

**Story 1.8 — Run History Sidebar**
- New `GET /eval/history` endpoint — returns all runs ordered by `created_at DESC`
- Filter parameters: `model` (string match in `models_selected` JSON), `engineer` (case-insensitive match in `engineer_names` JSON), `date_from` / `date_to` (ISO date strings)
- JSON field filtering done in Python (SQLite JSON function support is limited)
- Each history item includes: id, created_at (formatted "07 Apr 2026, 8:00pm"), modality, models_selected, engineer_names, run_label, status, error_message, winning_model
- Empty state: endpoint returns `{runs: [], total: 0}` when no runs exist (no error)
- Frontend sidebar: added Run History section below API keys with expandable filter bar
- Filter bar: model text input, engineer text input, date from/to date pickers, clear filters button
- Each run rendered as a clickable button showing status icon + label
- Clicking a completed run sets `last_run_id` in session_state → Results tab loads that run
- Sidebar auto-refreshes on every Streamlit rerun (triggered by polling loop or user action)
- Modality icons: 📄 text, 🖼️ image_text, 📊 structured_data
- Status icons: ✅ complete, ⏳ running, ❌ failed, 🔄 pending

**Schema additions**
- `backend/eval/schemas.py`: Added `EvalStatusResponse`, `EvalHistoryItem`, `EvalHistoryResponse`
- `backend/db/models.py`: Added `error_message`, `completed_at`, `engineer_names` to `EvalRun`

**Tests — 88/88 passing**
- `backend/tests/test_session_3.py` — 13 new tests:
  - Story 1.7 (7 tests): immediate pending creation, running/complete/failed status transitions, page refresh stability, run label persistence, auto-save
  - Story 1.8 (6 tests): empty state (isolated in-memory DB), ordering by recency, model filter, engineer filter, results load, failed run in history
- Sessions 1 + 2 fully backward compatible: 75 prior tests still passing

**Key Decisions Made This Session**
- BackgroundTasks (not asyncio.create_task): simpler, no async context issues, native to FastAPI, sufficient for MVP
- Static route `/eval/history` defined BEFORE dynamic `/eval/{run_id}/...` routes to prevent path parameter shadowing
- `StaticPool` required for in-memory SQLite test engine so all connections share the same database instance
- `engineer_names` stored as JSON list in EvalRun (not joined from prompts table) for efficient history filtering without joins
- datetime formatting is cross-platform (no `%-I` Linux-only flag): hour formatted manually in Python
- Programmatic Streamlit tab switching not supported — clicking a history run sets session_state.last_run_id and shows "Switch to Results tab" message

**Ideas for v1.1 (do not build)**
- Real-time run progress via SSE (per-prompt status updates while eval runs)
- Pagination for run history (currently returns all runs)
- Run comparison view — select two history entries side by side
- Export run history to CSV

**Next session:** Session 4 — Scoring UI & Verdict Display
