"""
eval/router.py — Session 1-7 endpoints.

Implements:
  POST /upload                   — parse CSV/JSONL/ZIP, validate fully (S1.6), return modality (S1.5)
  GET  /models/compatible        — return compatible/incompatible models for a modality (S1.5)
  POST /rubric/validate          — validate rubric weights
  POST /keys/validate            — validate API key format
  POST /eval/run                 — create run immediately with status=pending, execute via BackgroundTask (S1.7)
  GET  /eval/history             — list all runs ordered by recency with filters (S1.8)
  GET  /eval/compare             — compare two completed runs side-by-side (S2.6)
  GET  /eval/{run_id}/status     — poll run status (S1.7)
  GET  /eval/{run_id}/results    — fetch stored results
  GET  /eval/{run_id}/export/pdf — PDF export (S3.1)
  GET  /eval/{run_id}/export/json— JSON export (S3.2)
"""
import asyncio
import csv
import io
import json
import logging
import re
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
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
    get_mock_gt_score,
    get_mock_judge_scores,
    get_mock_response,
)

router = APIRouter()

_KEY_PATTERNS = re.compile(
    r"(sk-[A-Za-z0-9\-_]{10,}|AIza[A-Za-z0-9\-_]{10,}|sk-ant-[A-Za-z0-9\-_]{10,})",
    re.IGNORECASE,
)


def _sanitize_error(exc: Exception) -> str:
    """Strip API key patterns from exception messages before storing or returning."""
    return _KEY_PATTERNS.sub("[REDACTED]", str(exc))[:500]


def _get_user_id(x_user_id: str = Header(default="")) -> Optional[str]:
    """Read X-User-ID header. Returns None in DEV_MODE (auth bypass for local dev)."""
    if DEV_MODE:
        return None
    return x_user_id.strip() or None


def _assert_run_owner(run: EvalRun, user_id: Optional[str]) -> None:
    """Raise 403 if the run belongs to a different user. No-op in DEV_MODE."""
    if DEV_MODE or user_id is None:
        return
    if run.user_id and run.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied.")


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

        # Directory containing manifest.json — used to resolve relative image paths
        # e.g. if manifest is at "myfolder/manifest.json", images at "myfolder/images/foo.png"
        manifest_dir = manifest_names[0].rsplit("/", 1)[0] if "/" in manifest_names[0] else ""

        # Validate and extract image bytes into base64 data URIs
        # Support both "image" and "image_path" keys in manifest.json
        for row in rows:
            image_path = row.get("image_path") or row.get("image")
            if image_path:
                # Resolve path: try as-is first, then relative to manifest directory
                if image_path not in names and manifest_dir:
                    image_path = f"{manifest_dir}/{image_path}"
                if image_path not in names:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image file '{row.get('image_path') or row.get('image')}' referenced in manifest.json "
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


# ── ROUGE scoring helper ───────────────────────────────────────────────────


def _calc_rouge_scores(response_text: str, expected_output: str) -> tuple:
    """
    Calculate ROUGE-1 and ROUGE-L F1 scores using the rouge-score library.
    Pure Python, no API call. Returns (rouge_1_f1, rouge_l_f1) or (None, None) on error.
    """
    try:
        from rouge_score import rouge_scorer as _rs
        scorer = _rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(expected_output, response_text)
        return (
            round(scores["rouge1"].fmeasure, 4),
            round(scores["rougeL"].fmeasure, 4),
        )
    except Exception:
        return (None, None)


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
            run.error_message = _sanitize_error(exc)
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

            # Ground truth alignment (mock — only when expected_output present)
            gt_score, gt_reasoning = get_mock_gt_score(prompt_rec.expected_output)

            # ROUGE scores — pure Python, no API cost
            rouge_1, rouge_l = (
                _calc_rouge_scores(response_text, prompt_rec.expected_output)
                if prompt_rec.expected_output
                else (None, None)
            )

            # Mock judge token counts (representative of real judge call sizes)
            mock_judge_tin, mock_judge_tout = 500, 250
            mock_gt_tin, mock_gt_tout = 200, 50
            from backend.judge.judge import _calc_judge_cost
            mock_gt_calls = 1 if prompt_rec.expected_output else 0

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
                ground_truth_score=gt_score,
                ground_truth_reasoning=gt_reasoning,
                rouge_1_score=rouge_1,
                rouge_l_score=rouge_l,
                evidence_data={k: "[MOCK]" for k in dim_scores},
                eval_api_calls=1,
                judge_api_calls=1,
                gt_api_calls=mock_gt_calls,
                judge_tokens_in=mock_judge_tin,
                judge_tokens_out=mock_judge_tout,
                judge_cost_usd=_calc_judge_cost(mock_judge_tin, mock_judge_tout),
                gt_tokens_in=mock_gt_tin if mock_gt_calls else 0,
                gt_tokens_out=mock_gt_tout if mock_gt_calls else 0,
                gt_cost_usd=_calc_judge_cost(mock_gt_tin, mock_gt_tout) if mock_gt_calls else 0.0,
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
                "ground_truth_score": gt_score,
                "ground_truth_reasoning": gt_reasoning,
                "rouge_1_score": rouge_1,
                "rouge_l_score": rouge_l,
                "error": None,
            })

    db.commit()

    # Use real verdict generation so scoring logic is exercised even in dev mode
    generate_verdict(run_id, scored_results, rubric, db)

    from backend.verdict.verdict import save_variance_scores
    save_variance_scores(run_id, scored_results, rubric, db)


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

        # Step 2.5 — GT alignment scoring (only for results with expected_output)
        from backend.judge.judge import score_ground_truth_parallel
        has_gt = any(r.get("expected_output") for r in scored_results)
        if has_gt:
            logger.info("[run=%s] STEP 2.5 — scoring GT alignment for results with expected_output", run_id)
            scored_results = await score_ground_truth_parallel(scored_results, judge_api_key)
        else:
            for r in scored_results:
                r.setdefault("ground_truth_score", None)
                r.setdefault("ground_truth_reasoning", None)

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
            # ROUGE scores — pure Python, no API cost
            rouge_1, rouge_l = (
                _calc_rouge_scores(r.get("response_text", ""), r["expected_output"])
                if r.get("expected_output")
                else (None, None)
            )
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
                ground_truth_score=r.get("ground_truth_score"),
                ground_truth_reasoning=r.get("ground_truth_reasoning"),
                rouge_1_score=rouge_1,
                rouge_l_score=rouge_l,
                evidence_data=r.get("evidence") or {},
                model_error=_KEY_PATTERNS.sub("[REDACTED]", r.get("error") or "")[:500] or None,
                eval_api_calls=0 if r.get("error") else 1,
                judge_api_calls=r.get("judge_api_calls", 0),
                gt_api_calls=r.get("gt_api_calls", 0),
                judge_tokens_in=r.get("judge_tokens_in", 0),
                judge_tokens_out=r.get("judge_tokens_out", 0),
                judge_cost_usd=r.get("judge_cost_usd", 0.0),
                gt_tokens_in=r.get("gt_tokens_in", 0),
                gt_tokens_out=r.get("gt_tokens_out", 0),
                gt_cost_usd=r.get("gt_cost_usd", 0.0),
            )
            db.add(mr)
        db.commit()

        run.progress_pct = 90.0
        db.commit()

        # Step 4 — generate verdict
        logger.info("[run=%s] STEP 4 — generating verdict", run_id)
        generate_verdict(run_id, scored_results, rubric, db)

        # Step 5 — save per-prompt variance scores
        from backend.verdict.verdict import save_variance_scores
        save_variance_scores(run_id, scored_results, rubric, db)

        run.status = "complete"
        run.progress_pct = 100.0
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("[run=%s] Eval complete", run_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("[run=%s] Eval failed with unhandled exception: %s", run_id, exc)
        if run and db:
            run.status = "failed"
            run.error_message = _sanitize_error(exc)
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
    user_id: Optional[str] = Depends(_get_user_id),
):
    """
    Create an eval run immediately with status=pending, then execute via BackgroundTask.
    Returns run_id + status=pending before any model call happens.
    """
    run_id = str(uuid.uuid4())

    # Collect unique engineer names from prompt-level data
    engineer_names = list({p.engineer_name for p in request.prompts if p.engineer_name})

    # Detect modality from prompts — image_data present → image_text, else text
    modality = "image_text" if any(p.image_data for p in request.prompts) else "text"

    # Persist run with status=pending — this is the immediate commit
    run = EvalRun(
        id=run_id,
        created_at=datetime.now(timezone.utc),
        modality=modality,
        rubric_config=request.rubric.model_dump(),
        models_selected=request.models_selected,
        engineer_name=request.engineer_name,
        engineer_names=engineer_names or None,
        status="pending",
        custom_label=request.custom_label,
        user_id=user_id,
    )
    db.add(run)
    db.commit()

    # Persist prompts (image_data stored for image_text modality)
    for idx, p in enumerate(request.prompts):
        logger.info(
            "[run=%s] Prompt %d image_data present=%s len=%s",
            run_id, idx, bool(p.image_data), len(p.image_data) if p.image_data else 0,
        )
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


# ── Story 2.6 — Run comparison helpers ────────────────────────────────────

_COMPARE_DIMS = ["accuracy", "hallucination", "instruction_following", "conciseness"]


def _calc_run_scores_and_costs(results) -> tuple[dict, dict]:
    """
    Return (scores_by_model, costs_by_model) from a list of ModelResult ORM rows.
    scores_by_model: {model: {dim: avg_score}}
    costs_by_model:  {model: total_cost_usd}
    """
    from collections import defaultdict
    dim_sum: dict = defaultdict(lambda: defaultdict(float))
    dim_cnt: dict = defaultdict(lambda: defaultdict(int))
    cost_sum: dict = defaultdict(float)

    for r in results:
        m = r.model_name
        cost_sum[m] += r.cost_usd or 0.0
        for dim in _COMPARE_DIMS:
            v = (r.dimension_scores or {}).get(dim)
            if v is not None:
                dim_sum[m][dim] += float(v)
                dim_cnt[m][dim] += 1

    scores = {
        m: {
            dim: round(dim_sum[m][dim] / dim_cnt[m][dim], 3) if dim_cnt[m][dim] > 0 else None
            for dim in _COMPARE_DIMS
        }
        for m in dim_sum
    }
    costs = {m: round(cost_sum[m], 6) for m in cost_sum}
    return scores, costs


def _generate_compare_insight(score_delta: dict, shared_models: set) -> str:
    """One-sentence insight derived from score deltas across shared models."""
    from collections import defaultdict
    if not shared_models:
        return (
            "No shared models between runs — comparison shows different model configurations."
        )

    dim_deltas: dict = defaultdict(list)
    for m in shared_models:
        if m in score_delta:
            for dim in _COMPARE_DIMS:
                d = score_delta[m].get(dim)
                if d is not None:
                    dim_deltas[dim].append(d)

    if not dim_deltas:
        return (
            "Scores were consistent across both runs — model performance is stable on this dataset."
        )

    best_dim = max(dim_deltas, key=lambda d: sum(dim_deltas[d]) / len(dim_deltas[d]))
    best_avg = sum(dim_deltas[best_dim]) / len(dim_deltas[best_dim])

    if best_avg > 0.1:
        dim_display = best_dim.replace("_", " ")
        return (
            f"{dim_display.capitalize()} improved by {best_avg:.1f} points between runs, "
            f"suggesting prompt refinements had a positive impact on {dim_display} responses."
        )
    return "Scores were consistent across both runs — model performance is stable on this dataset."


# ── Story 1.8 — Run history (static route BEFORE dynamic {run_id} routes) ─


@router.get("/eval/history", response_model=EvalHistoryResponse)
def get_eval_history(
    model: Optional[str] = None,
    engineer: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(_get_user_id),
):
    """
    Return eval runs for the requesting user, ordered most-recent first.
    In DEV_MODE returns all runs. In prod filters by X-User-ID header.
    """
    query = db.query(EvalRun).order_by(EvalRun.created_at.desc())

    # Scope to the authenticated user's runs
    if user_id:
        query = query.filter(EvalRun.user_id == user_id)

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


# ── Story 2.6 — Run comparison (static route BEFORE dynamic {run_id} routes) ─


@router.get("/eval/compare")
def compare_runs(
    run_a: str,
    run_b: str,
    db: Session = Depends(get_db),
):
    """
    Compare two completed eval runs side-by-side.
    Returns per-model dimension scores, costs, deltas, and an auto-insight sentence.
    Models present in only one run show null deltas (not an error).
    """
    run_a_obj = db.query(EvalRun).filter(EvalRun.id == run_a).first()
    run_b_obj = db.query(EvalRun).filter(EvalRun.id == run_b).first()

    if not run_a_obj:
        raise HTTPException(status_code=404, detail=f"Run '{run_a}' not found.")
    if not run_b_obj:
        raise HTTPException(status_code=404, detail=f"Run '{run_b}' not found.")

    if run_a_obj.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Run A ('{run_a}') is not complete (status: {run_a_obj.status}). "
                   "Both runs must be completed to compare.",
        )
    if run_b_obj.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Run B ('{run_b}') is not complete (status: {run_b_obj.status}). "
                   "Both runs must be completed to compare.",
        )

    verdict_a = db.query(Verdict).filter(Verdict.eval_run_id == run_a).first()
    verdict_b = db.query(Verdict).filter(Verdict.eval_run_id == run_b).first()

    results_a = db.query(ModelResult).filter(ModelResult.eval_run_id == run_a).all()
    results_b = db.query(ModelResult).filter(ModelResult.eval_run_id == run_b).all()

    scores_a, costs_a = _calc_run_scores_and_costs(results_a)
    scores_b, costs_b = _calc_run_scores_and_costs(results_b)

    models_a = set(scores_a.keys())
    models_b = set(scores_b.keys())
    all_models = models_a | models_b
    shared_models = models_a & models_b

    # Compute per-model deltas (None for models missing from one run)
    score_delta: dict = {}
    cost_delta: dict = {}
    for m in all_models:
        if m in scores_a and m in scores_b:
            score_delta[m] = {
                dim: (
                    round(scores_b[m][dim] - scores_a[m][dim], 3)
                    if scores_a[m].get(dim) is not None and scores_b[m].get(dim) is not None
                    else None
                )
                for dim in _COMPARE_DIMS
            }
            cost_delta[m] = round(costs_b.get(m, 0.0) - costs_a.get(m, 0.0), 6)
        else:
            score_delta[m] = {dim: None for dim in _COMPARE_DIMS}
            cost_delta[m] = None

    winner_a = verdict_a.winning_model if verdict_a else None
    winner_b = verdict_b.winning_model if verdict_b else None
    winner_changed = winner_a != winner_b

    insight = _generate_compare_insight(score_delta, shared_models)

    return {
        "run_a": {
            "id": run_a,
            "label": run_a_obj.custom_label or f"Run {run_a[:8]}",
            "date": _format_run_datetime(run_a_obj.created_at),
            "models": run_a_obj.models_selected or [],
            "winner": winner_a,
            "scores": scores_a,
            "costs": costs_a,
        },
        "run_b": {
            "id": run_b,
            "label": run_b_obj.custom_label or f"Run {run_b[:8]}",
            "date": _format_run_datetime(run_b_obj.created_at),
            "models": run_b_obj.models_selected or [],
            "winner": winner_b,
            "scores": scores_b,
            "costs": costs_b,
        },
        "deltas": {
            "score_delta": score_delta,
            "cost_delta": cost_delta,
            "verdict_changed": winner_changed,
            "winner_changed": winner_changed,
            "insight": insight,
        },
    }


# ── Story 3.1 — PDF export (static sub-route BEFORE {run_id} dynamic routes) ─


@router.get("/eval/{run_id}/export/pdf")
def export_pdf(run_id: str, db: Session = Depends(get_db), user_id: Optional[str] = Depends(_get_user_id)):
    """
    Generate and return a PDF evaluation report for a completed run.
    Max 2 pages. Per-prompt section shows top 5 prompts by variance score.
    """
    from fastapi.responses import Response
    from backend.export.pdf_exporter import generate_pdf_bytes

    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    _assert_run_owner(run, user_id)
    if run.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not complete (status: {run.status}). "
                   "PDF export is only available for completed runs.",
        )

    try:
        pdf_bytes = generate_pdf_bytes(run_id, db)
    except Exception as exc:
        logger.exception("[run=%s] PDF export failed: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}") from exc

    label = (run.custom_label or run_id[:8]).replace(" ", "_")
    filename = f"verdictai_{label}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Story 3.2 — JSON export ────────────────────────────────────────────────


@router.get("/eval/{run_id}/export/json")
def export_json(run_id: str, db: Session = Depends(get_db), user_id: Optional[str] = Depends(_get_user_id)):
    """
    Generate and return the canonical JSON export for a completed eval run.
    Content-Disposition filename: verdictai_{label}_{date}.json
    """
    from fastapi.responses import JSONResponse
    from backend.export.json_exporter import generate_json_report

    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    _assert_run_owner(run, user_id)
    if run.status != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not complete (status: {run.status}). "
                   "JSON export is only available for completed runs.",
        )

    try:
        payload = generate_json_report(run_id, db)
    except Exception as exc:
        logger.exception("[run=%s] JSON export failed: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"JSON export failed: {exc}") from exc

    label = (run.custom_label or run_id[:8]).replace(" ", "_")
    date_str = run.created_at.strftime("%Y%m%d") if run.created_at else "unknown"
    filename = f"verdictai_{label}_{date_str}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Image serving (static sub-sub-route BEFORE {run_id} dynamic routes) ──────


@router.get("/eval/{run_id}/image/{prompt_index}")
def get_prompt_image(run_id: str, prompt_index: int, db: Session = Depends(get_db)):
    """
    Return the raw image bytes for the given prompt in an image_text run.
    Decodes the base64 data URI stored in the Prompt record.
    Returns 404 if the run, prompt, or image data are not found.
    """
    import base64 as _base64
    from fastapi.responses import Response as _Response

    # Locate any ModelResult for this run + prompt_index to get the prompt_id
    result = (
        db.query(ModelResult)
        .filter(
            ModelResult.eval_run_id == run_id,
            ModelResult.prompt_index == str(prompt_index),
        )
        .first()
    )
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt index {prompt_index} not found for run '{run_id}'.",
        )

    prompt = db.query(Prompt).filter(Prompt.id == result.prompt_id).first()
    if not prompt or not prompt.image_data:
        raise HTTPException(
            status_code=404,
            detail=f"No image data for prompt {prompt_index} in run '{run_id}'.",
        )

    # Decode "data:{mime};base64,{b64}" → raw bytes
    data_uri = prompt.image_data
    if not data_uri.startswith("data:"):
        raise HTTPException(status_code=500, detail="Stored image data has unexpected format.")

    try:
        header, b64_content = data_uri.split(",", 1)
        mime = header.split(":")[1].split(";")[0]   # e.g. "image/jpeg"
        image_bytes = _base64.b64decode(b64_content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to decode image: {exc}") from exc

    return _Response(content=image_bytes, media_type=mime)


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
        progress_pct=run.progress_pct,
    )


# ── Eval results ───────────────────────────────────────────────────────────


@router.get("/eval/{run_id}/results", response_model=EvalResultsResponse)
def get_eval_results(run_id: str, db: Session = Depends(get_db), user_id: Optional[str] = Depends(_get_user_id)):
    """Fetch stored results for a completed eval run."""
    run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")
    _assert_run_owner(run, user_id)

    db_results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
    db_verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()

    # Build prompt_text lookup (join by prompt_id)
    prompt_texts: dict = {
        p.id: p.prompt_text
        for p in db.query(Prompt).filter(Prompt.eval_run_id == run_id).all()
    }

    results_out = [
        ModelResultOut(
            model_name=r.model_name,
            prompt_index=int(r.prompt_index),
            prompt_text=prompt_texts.get(r.prompt_id),
            response_text=r.response_text,
            dimension_scores=r.dimension_scores,
            dimension_reasoning=r.dimension_reasoning,
            hallucination_flagged=r.hallucination_flagged,
            tokens_used=r.tokens_used,
            tokens_in=r.tokens_in,
            tokens_out=r.tokens_out,
            cost_usd=r.cost_usd,
            variance_score=r.variance_score,
            ground_truth_score=r.ground_truth_score,
            ground_truth_reasoning=r.ground_truth_reasoning,
            rouge_1_score=r.rouge_1_score,
            rouge_l_score=r.rouge_l_score,
            model_error=r.model_error,
            eval_api_calls=r.eval_api_calls,
            judge_api_calls=r.judge_api_calls,
            gt_api_calls=r.gt_api_calls,
            judge_tokens_in=r.judge_tokens_in,
            judge_tokens_out=r.judge_tokens_out,
            judge_cost_usd=r.judge_cost_usd,
            gt_tokens_in=r.gt_tokens_in,
            gt_tokens_out=r.gt_tokens_out,
            gt_cost_usd=r.gt_cost_usd,
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
        modality=run.modality or "text",
        results=results_out,
        verdict=verdict_out,
    )
