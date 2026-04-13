"""
eval/router.py — Session 1-4 endpoints.

Implements:
  POST /upload                  — parse CSV/JSONL/ZIP, validate fully (S1.6), return modality (S1.5)
  GET  /models/compatible       — return compatible/incompatible models for a modality (S1.5)
  POST /rubric/validate         — validate rubric weights
  POST /keys/validate           — validate API key format
  POST /eval/run                — create run immediately with status=pending, execute via BackgroundTask (S1.7)
  GET  /eval/history            — list all runs ordered by recency with filters (S1.8)
  GET  /eval/{run_id}/status    — poll run status (S1.7)
  GET  /eval/{run_id}/results   — fetch stored results
"""
import asyncio
import csv
import io
import json
import logging
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import DEV_MODE, MAX_PROMPTS, MIN_PROMPTS, MODELS_CONFIG

logger = logging.getLogger(__name__)
from backend.db.database import SessionLocal, get_db
from backend.db.models import EvalRun, ModelResult, Prompt, Verdict
from backend.eval.schemas import (
    APIKeys,
    EvalHistoryItem,
    EvalHistoryResponse,
    EvalResultsResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalStatusResponse,
    IncompatibleModel,
    KeyValidationResponse,
    ModelResultOut,
    ModelsCompatibleResponse,
    PromptInput,
    RubricWeights,
    UploadResponse,
    ValidationWarning,
)
from backend.judge.mock_judge import (
    calculate_cost_efficiency,
    calculate_mock_cost,
    get_mock_judge_scores,
    get_mock_response,
)

router = APIRouter()

# Constants for validation
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MODALITY_MATRIX = MODELS_CONFIG.get("modality_matrix", {})
PLATFORM_SUGGESTIONS = MODELS_CONFIG.get("platform_suggestions", {})


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_csv(content: bytes) -> list[dict]:
    """Parse CSV and return list of row dicts."""
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader) if reader else []


def _parse_jsonl(content: bytes) -> list[dict]:
    """Parse JSONL and return list of row dicts."""
    lines = content.decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _detect_modality_from_jsonl(rows: list[dict]) -> str:
    """Detect modality from JSONL: text, structured_data, or image_text."""
    if not rows:
        return "text"
    # Check for 'data' field (structured data)
    if any("data" in row for row in rows):
        return "structured_data"
    # Default
    return "text"


def _parse_zip(content: bytes) -> tuple[list[dict], str, list[ValidationWarning]]:
    """
    Parse ZIP containing manifest.json + images folder.
    Returns (rows, modality, warnings).
    Validates all referenced image files exist, extracts bytes as base64 data URIs.
    """
    import base64 as _base64

    MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    warnings: list[ValidationWarning] = []

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()

        # Find manifest.json
        manifest_names = [n for n in names if n.endswith("manifest.json")]
        if not manifest_names:
            raise HTTPException(
                status_code=422,
                detail=(
                    "ZIP must contain a manifest.json file. "
                    "Expected: manifest.json at root or in any folder."
                ),
            )

        manifest_data = json.loads(z.read(manifest_names[0]))
        rows = manifest_data if isinstance(manifest_data, list) else manifest_data.get("prompts", [])

        # Validate and extract image bytes into base64 data URIs
        for row in rows:
            if "image_path" in row and row["image_path"]:
                image_path = row["image_path"]
                if image_path not in names:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image file '{image_path}' referenced in manifest.json "
                            f"not found in ZIP. Check the file path and ensure all "
                            f"referenced images are included."
                        ),
                    )
                ext = "." + image_path.split(".")[-1].lower() if "." in image_path else ""
                if ext not in VALID_IMAGE_EXTENSIONS:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image file '{image_path}' has unsupported format '{ext}'. "
                            f"Supported formats: jpg, jpeg, png, webp"
                        ),
                    )
                # Extract bytes and encode as base64 data URI
                image_bytes = z.read(image_path)
                mime = MIME_MAP.get(ext, "image/jpeg")
                b64 = _base64.b64encode(image_bytes).decode("utf-8")
                row["image_data"] = f"data:{mime};base64,{b64}"

    return rows, "image_text", warnings


def _detect_modality(rows: list[dict], file_type: str) -> str:
    """Detect modality: text, image_text, or structured_data."""
    if file_type == "zip":
        return "image_text"  # ZIP uploads are always image_text (validated above)

    if file_type == "jsonl":
        return _detect_modality_from_jsonl(rows)

    # CSV is always text
    return "text"


def _rows_to_prompts(rows: list[dict]) -> tuple[list[PromptInput], list[ValidationWarning]]:
    """
    Convert rows to PromptInput objects with validation.
    Returns (prompts, warnings).
    """
    warnings: list[ValidationWarning] = []
    prompts = []

    for i, row in enumerate(rows, start=1):
        # Check required 'prompt' column
        if "prompt" not in row or not str(row.get("prompt", "")).strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Row {i}: missing or empty 'prompt' column. "
                    f"Every prompt must have a non-empty 'prompt' field."
                ),
            )

        # Warn if missing optional fields
        if not row.get("expected_output"):
            warnings.append(ValidationWarning(
                field=f"row_{i}_expected_output",
                message=f"Row {i}: missing 'expected_output'. Ground truth comparison disabled for this prompt."
            ))

        if not row.get("engineer_name"):
            warnings.append(ValidationWarning(
                field=f"row_{i}_engineer_name",
                message=f"Row {i}: missing 'engineer_name'. This prompt won't be attributed to a team member."
            ))

        prompts.append(PromptInput(
            prompt=str(row["prompt"]).strip(),
            expected_output=row.get("expected_output") or None,
            engineer_name=row.get("engineer_name") or None,
            image_data=row.get("image_data") or None,
        ))

    return prompts, warnings


def _validate_prompt_count(count: int) -> None:
    """Validate prompt count is between MIN and MAX."""
    if count == 0:
        raise HTTPException(
            status_code=422,
            detail="Dataset is empty. Upload at least 5 prompts.",
        )
    if count < MIN_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset has {count} prompts but minimum is {MIN_PROMPTS}. "
                f"Add at least {MIN_PROMPTS - count} more prompts."
            ),
        )
    if count > MAX_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset has {count} prompts but maximum is {MAX_PROMPTS}. "
                f"Remove {count - MAX_PROMPTS} prompts to proceed."
            ),
        )


def _make_validation_summary(
    prompt_count: int,
    modality: str,
    has_gt: bool,
    gt_count: int,
    has_engineers: bool,
    engineer_count: int,
) -> str:
    """Generate a human-readable validation summary."""
    gt_text = f"✓ {gt_count}/{prompt_count}" if has_gt else "✗ 0 provided"
    eng_text = f"✓ {engineer_count} engineers" if has_engineers else "✗ 0 tagged"
    return (
        f"✓ **{prompt_count} prompts detected** | "
        f"**Modality:** {modality} | "
        f"**Ground truth:** {gt_text} | "
        f"**Engineers:** {eng_text}"
    )


def _format_run_datetime(dt: Optional[datetime]) -> str:
    """Format datetime as '07 Apr 2026, 8:00pm' (cross-platform)."""
    if not dt:
        return ""
    hour = dt.hour
    minute = dt.strftime("%M")
    ampm = "am" if hour < 12 else "pm"
    hour_12 = hour % 12 or 12
    return dt.strftime(f"%d %b %Y, {hour_12}:{minute}{ampm}")


# ── Story 1.7/2.1/2.2 — Background eval execution ──────────────────────────


def _execute_eval(run_id: str, request: EvalRunRequest) -> None:
    """
    Background task: execute the eval and update run status.
    Creates its own DB session — the request session is already closed.
    Status flow: pending → running → complete (or failed on exception).

    DEV_MODE=true:  mock responses + mock judge scores (no API calls)
    DEV_MODE=false: real runner (asyncio) + real judge + real verdict
    """
    db = SessionLocal()
    run = None
    try:
        run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
        if not run:
            return

        run.status = "running"
        run.progress_pct = 0.0
        db.commit()

        rubric = request.rubric.model_dump()
        prompt_records = db.query(Prompt).filter(Prompt.eval_run_id == run_id).all()

        if DEV_MODE:
            _run_mock_eval(run_id, request, prompt_records, rubric, db)
        else:
            # Real async path — close this session first; runner opens its own
            db.close()
            db = None
            run = None
            asyncio.run(_run_real_eval_async(run_id, request))
            return  # async function manages all status transitions

        run.status = "complete"
        run.completed_at = datetime.now(timezone.utc)
        run.progress_pct = 100.0
        db.commit()

    except Exception as exc:  # noqa: BLE001
        if run and db:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        if db:
            db.close()


def _run_mock_eval(
    run_id: str,
    request: EvalRunRequest,
    prompt_records: list,
    rubric: dict,
    db,
) -> None:
    """
    DEV_MODE mock execution path.
    Uses mock responses and judge scores from models.yaml.
    Full verdict generation runs through real verdict.py logic.
    """
    from backend.verdict.verdict import generate_verdict

    scored_results: list[dict] = []

    for idx, prompt_rec in enumerate(prompt_records):
        for model_id in request.models_selected:
            response_text = get_mock_response(model_id)
            judge_out = get_mock_judge_scores()
            scores = judge_out["scores"]
            reasoning = judge_out["reasoning"]

            tokens_in, tokens_out = 120, 85
            tokens = {"input": tokens_in, "output": tokens_out}
            cost_usd = calculate_mock_cost(model_id, tokens)

            # Session 4 semantics: hallucination 10=good, 0=bad; flagged when <= 3
            hallucination_flagged = scores.get("hallucination", 10) <= 3
            hallucination_reason = reasoning.get("hallucination") if hallucination_flagged else None

            # cost_efficiency is auto-calculated in verdict.py — keep it out of dim_scores
            dim_scores = {k: v for k, v in scores.items() if k != "cost_efficiency"}

            result = ModelResult(
                id=str(uuid.uuid4()),
                eval_run_id=run_id,
                prompt_id=prompt_rec.id,
                prompt_index=str(idx),
                model_name=model_id,
                response_text=response_text,
                dimension_scores=dim_scores,
                dimension_reasoning=reasoning,
                hallucination_flagged=hallucination_flagged,
                hallucination_reason=hallucination_reason,
                tokens_used=tokens,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost_usd,
            )
            db.add(result)

            scored_results.append({
                "model_id": model_id,
                "prompt_id": prompt_rec.id,
                "prompt_index": idx,
                "prompt_text": prompt_rec.prompt_text,
                "expected_output": prompt_rec.expected_output,
                "response_text": response_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
                "scores": dim_scores,
                "reasoning": reasoning,
                "evidence": {k: "[MOCK]" for k in dim_scores},
                "hallucination_flagged": hallucination_flagged,
                "hallucination_reason": hallucination_reason,
                "error": None,
            })

    db.commit()

    # Use real verdict generation so scoring logic is exercised even in dev mode
    generate_verdict(run_id, scored_results, rubric, db)


async def _run_real_eval_async(run_id: str, request: EvalRunRequest) -> None:
    """
    DEV_MODE=false async execution path.
    Runs runner → judge → verdict → saves all results.
    Manages its own DB session.
    """
    from backend.runner.runner import run_models_parallel
    from backend.judge.judge import score_responses_parallel
    from backend.verdict.verdict import generate_verdict

    db = SessionLocal()
    run = None
    try:
        run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
        if not run:
            logger.error("[run=%s] EvalRun not found in DB", run_id)
            return

        prompt_records = db.query(Prompt).filter(Prompt.eval_run_id == run_id).all()
        logger.info(
            "[run=%s] Starting real eval — models=%s prompts=%d",
            run_id, request.models_selected, len(prompt_records),
        )

        prompts_data = [
            {
                "prompt_id": p.id,
                "prompt_text": p.prompt_text,
                "image_data": p.image_data,
                "expected_output": p.expected_output,
                "prompt_index": i,
            }
            for i, p in enumerate(prompt_records)
        ]

        api_keys = request.api_keys.model_dump()
        rubric = request.rubric.model_dump()

        openai_key = api_keys.get("openai_api_key", "")
        logger.info(
            "[run=%s] API key present: openai=%s anthropic=%s google=%s",
            run_id,
            bool(openai_key and openai_key.strip()),
            bool(api_keys.get("anthropic_api_key")),
            bool(api_keys.get("google_api_key")),
        )

        # Step 1 — run all models × prompts in parallel
        logger.info("[run=%s] STEP 1 — calling runner (model × prompt matrix)", run_id)
        model_results = await run_models_parallel(request.models_selected, prompts_data, api_keys)

        failed = [r for r in model_results if r.get("error")]
        succeeded = [r for r in model_results if not r.get("error")]
        logger.info(
            "[run=%s] Runner done — %d succeeded, %d failed",
            run_id, len(succeeded), len(failed),
        )
        for r in failed:
            logger.error(
                "[run=%s] Runner ERROR model=%s prompt_idx=%s: %s",
                run_id, r["model_id"], r["prompt_index"], r["error"],
            )

        run.progress_pct = 50.0
        db.commit()

        # Step 2 — judge all results in parallel
        judge_api_key = openai_key
        logger.info("[run=%s] STEP 2 — calling judge on %d results", run_id, len(succeeded))
        scored_results = await score_responses_parallel(model_results, rubric, judge_api_key)

        judged = [r for r in scored_results if r.get("scores") and any(v is not None for v in r["scores"].values())]
        logger.info(
            "[run=%s] Judge done — %d/%d results have scores",
            run_id, len(judged), len(scored_results),
        )
        for r in scored_results:
            if r.get("judge_error"):
                logger.error(
                    "[run=%s] Judge returned null scores for model=%s prompt_idx=%s",
                    run_id, r["model_id"], r["prompt_index"],
                )

        # Step 3 — save ModelResult rows
        logger.info("[run=%s] STEP 3 — saving %d ModelResult rows", run_id, len(scored_results))
        for r in scored_results:
            dim_scores = r.get("scores", {})
            mr = ModelResult(
                id=str(uuid.uuid4()),
                eval_run_id=run_id,
                prompt_id=r["prompt_id"],
                prompt_index=str(r["prompt_index"]),
                model_name=r["model_id"],
                response_text=r.get("response_text", ""),
                dimension_scores=dim_scores,
                dimension_reasoning=r.get("reasoning", {}),
                hallucination_flagged=r.get("hallucination_flagged", False),
                hallucination_reason=r.get("hallucination_reason"),
                tokens_used={"input": r["tokens_in"], "output": r["tokens_out"]},
                tokens_in=r["tokens_in"],
                tokens_out=r["tokens_out"],
                cost_usd=r.get("cost_usd", 0.0),
            )
            db.add(mr)
        db.commit()

        run.progress_pct = 90.0
        db.commit()

        # Step 4 — generate verdict
        logger.info("[run=%s] STEP 4 — generating verdict", run_id)
        generate_verdict(run_id, scored_results, rubric, db)

        run.status = "complete"
        run.progress_pct = 100.0
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("[run=%s] Eval complete", run_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("[run=%s] Eval failed with unhandled exception: %s", run_id, exc)
        if run and db:
            run.status = "failed"
            run.error_message = str(exc)[:500]
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        if db:
            db.close()


# ── Story 1.6 — Upload with Full Validation ────────────────────────────────


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload and validate dataset. Full validation includes:
    - CSV/JSONL/ZIP format checking
    - Required 'prompt' column
    - Prompt count (5-100)
    - ZIP: manifest.json + valid image files
    - Modality detection
    """
    filename = (file.filename or "").lower()
    content = await file.read()

    # Detect file type and parse
    if filename.endswith(".csv"):
        rows = _parse_csv(content)
        file_type = "csv"
    elif filename.endswith(".jsonl"):
        rows = _parse_jsonl(content)
        file_type = "jsonl"
    elif filename.endswith(".zip"):
        rows, _, zip_warnings = _parse_zip(content)
        file_type = "zip"
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Upload .csv, .jsonl, or .zip",
        )

    # Validate prompt count
    _validate_prompt_count(len(rows))

    # Parse rows to PromptInput with warnings
    prompts, validation_warnings = _rows_to_prompts(rows)

    # Merge ZIP warnings if any
    if file_type == "zip" and "zip_warnings" in locals():
        validation_warnings.extend(zip_warnings)

    # Detect modality
    modality = _detect_modality(rows, file_type)

    # Count ground truth and engineer names
    gt_count = sum(1 for p in prompts if p.expected_output)
    has_gt = gt_count > 0
    eng_count = sum(1 for p in prompts if p.engineer_name)
    has_engineers = eng_count > 0

    # Generate validation summary
    summary = _make_validation_summary(
        len(prompts), modality, has_gt, gt_count, has_engineers, eng_count
    )

    return UploadResponse(
        prompt_count=len(prompts),
        has_ground_truth=has_gt,
        has_engineer_names=has_engineers,
        modality=modality,
        prompts=prompts,
        warnings=validation_warnings,
        validation_summary=summary,
    )


# ── Story 1.5 — Modality-based Model Compatibility ────────────────────────


@router.get("/models/compatible", response_model=ModelsCompatibleResponse)
def get_compatible_models(modality: str = "text"):
    """
    Return models compatible with a given modality.
    Reads modality_matrix and platform_suggestions from models.yaml.
    """
    if modality not in MODALITY_MATRIX:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown modality '{modality}'. Valid: text, image_text, structured_data, video, audio",
        )

    modality_info = MODALITY_MATRIX[modality]
    compatible_ids: list[str] = modality_info.get("mvp_compatible_models", [])

    # Get full model details for compatible models
    all_models = {m["id"]: m for m in MODELS_CONFIG.get("mvp_models", [])}
    compatible_models = [all_models[mid] for mid in compatible_ids if mid in all_models]

    # Build incompatible list (all MVP models not in compatible)
    all_mvp_ids = {m["id"] for m in MODELS_CONFIG.get("mvp_models", [])}
    incompatible_ids = all_mvp_ids - set(compatible_ids)

    incompatible_models = [
        IncompatibleModel(
            model=mid,
            reason=MODALITY_MATRIX[modality].get("incompatible_models", {}).get(
                mid, f"Not compatible with {modality}"
            ),
        )
        for mid in incompatible_ids
    ]

    # Build suggestions from models.yaml platform_suggestions
    suggestions = {}
    for model_id, suggestion_info in PLATFORM_SUGGESTIONS.items():
        if suggestion_info.get("incompatible_with") == modality:
            suggestions[model_id] = {
                "suggest": suggestion_info.get("suggest"),
                "message": suggestion_info.get("message"),
            }

    return ModelsCompatibleResponse(
        modality=modality,
        compatible_models=compatible_models,
        incompatible_models=incompatible_models,
        suggestions=suggestions,
    )


# ── Story 1.2 — API Keys ───────────────────────────────────────────────────


@router.post("/keys/validate", response_model=KeyValidationResponse)
def validate_keys(keys: APIKeys):
    """Validate API key format. Keys are never persisted."""
    return KeyValidationResponse(
        valid=True,
        openai=True,
        anthropic=keys.anthropic_api_key is not None,
        google=keys.google_api_key is not None,
    )


# ── Story 1.3 — Rubric ─────────────────────────────────────────────────────


@router.post("/rubric/validate", response_model=RubricWeights)
def validate_rubric(rubric: RubricWeights):
    """Validate rubric weights. Returns the validated rubric."""
    return rubric


# ── Story 1.7 — Eval run with background execution ────────────────────────

# NOTE: /eval/history MUST be defined before /eval/{run_id}/... routes.
# FastAPI matches static paths before path parameters, but being explicit is safer.


@router.post("/eval/run", response_model=EvalRunResponse)
def start_eval_run(
    request: EvalRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Create an eval run immediately with status=pending, then execute via BackgroundTask.
    Returns run_id + status=pending before any model call happens.
    """
    run_id = str(uuid.uuid4())

    # Collect unique engineer names from prompt-level data
    engineer_names = list({p.engineer_name for p in request.prompts if p.engineer_name})

    # Persist run with status=pending — this is the immediate commit
    run = EvalRun(
        id=run_id,
        created_at=datetime.now(timezone.utc),
        modality="text",  # client provides modality; default text for backward compat
        rubric_config=request.rubric.model_dump(),
        models_selected=request.models_selected,
        engineer_name=request.engineer_name,
        engineer_names=engineer_names or None,
        status="pending",
        custom_label=request.custom_label,
    )
    db.add(run)
    db.commit()

    # Persist prompts (image_data stored for image_text modality)
    for p in request.prompts:
        prompt_rec = Prompt(
            id=str(uuid.uuid4()),
            eval_run_id=run_id,
            prompt_text=p.prompt,
            expected_output=p.expected_output,
            engineer_name=p.engineer_name,
            image_data=p.image_data,
        )
        db.add(prompt_rec)
    db.commit()

    # Schedule eval execution as background task — returns immediately
    background_tasks.add_task(_execute_eval, run_id, request)

    return EvalRunResponse(run_id=run_id, status="pending")


# ── Story 1.8 — Run history (static route BEFORE dynamic {run_id} routes) ─


@router.get("/eval/history", response_model=EvalHistoryResponse)
def get_eval_history(
    model: Optional[str] = None,
    engineer: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Return all eval runs ordered most-recent first.
    Filters: model (string match in models_selected JSON),
             engineer (case-insensitive match in engineer_names JSON),
             date_from / date_to (ISO date strings YYYY-MM-DD).
    """
    query = db.query(EvalRun).order_by(EvalRun.created_at.desc())

    # Date filters — applied in SQL
    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from)
            query = query.filter(EvalRun.created_at >= dt_from)
        except ValueError:
            pass
    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to)
            query = query.filter(EvalRun.created_at <= dt_to)
        except ValueError:
            pass

    runs = query.all()

    # Model + engineer filters — applied in Python (SQLite JSON support is limited)
    if model:
        runs = [r for r in runs if model in (r.models_selected or [])]
    if engineer:
        runs = [
            r for r in runs
            if any(engineer.lower() in (e or "").lower() for e in (r.engineer_names or []))
        ]

    # Build history items — join with verdicts for winning model
    items: list[EvalHistoryItem] = []
    for run in runs:
        verdict = db.query(Verdict).filter(Verdict.eval_run_id == run.id).first()
        items.append(EvalHistoryItem(
            id=run.id,
            created_at=_format_run_datetime(run.created_at),
            modality=run.modality,
            models_selected=run.models_selected or [],
            engineer_names=run.engineer_names or [],
            run_label=run.custom_label,
            status=run.status,
            error_message=run.error_message,
            winning_model=verdict.winning_model if verdict else None,
            overall_score=None,  # full scoring in Session 4
        ))

    return EvalHistoryResponse(runs=items, total=len(items))


# ── Story 1.7 — Status polling ─────────────────────────────────────────────


@router.get("/eval/{run_id}/status", response_model=EvalStatusResponse)
def get_eval_status(run_id: str, db: Session = Depends(get_db)):
    """Poll the current status of an eval run. Used by frontend every 2s while running."""
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")

    return EvalStatusResponse(
        run_id=run_id,
        status=run.status,
        error_message=run.error_message,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


# ── Eval results ───────────────────────────────────────────────────────────


@router.get("/eval/{run_id}/results", response_model=EvalResultsResponse)
def get_eval_results(run_id: str, db: Session = Depends(get_db)):
    """Fetch stored results for a completed eval run."""
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")

    db_results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
    db_verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()

    results_out = [
        ModelResultOut(
            model_name=r.model_name,
            prompt_index=int(r.prompt_index),
            response_text=r.response_text,
            dimension_scores=r.dimension_scores,
            dimension_reasoning=r.dimension_reasoning,
            hallucination_flagged=r.hallucination_flagged,
            tokens_used=r.tokens_used,
            cost_usd=r.cost_usd,
        )
        for r in db_results
    ]

    verdict_out = None
    if db_verdict:
        verdict_out = {
            "winning_model": db_verdict.winning_model,
            "summary": db_verdict.summary,
            "score_breakdown": db_verdict.score_breakdown,
            "cost_comparison": db_verdict.cost_comparison,
            "hallucination_warnings": db_verdict.hallucination_warnings,
        }

    return EvalResultsResponse(
        run_id=run_id,
        status=run.status,
        results=results_out,
        verdict=verdict_out,
    )
