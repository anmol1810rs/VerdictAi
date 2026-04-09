"""
eval/router.py — Session 2 endpoints (Stories 1.4, 1.5, 1.6).

Implements:
  POST /upload                  — parse CSV/JSONL/ZIP, validate fully (S1.6), return modality (S1.5)
  GET  /models/compatible       — return compatible/incompatible models for a modality (S1.5)
  POST /rubric/validate         — validate rubric weights
  POST /keys/validate           — validate API key format
  POST /eval/run                — create run, execute mock judge, persist to DB
  GET  /eval/{run_id}/results   — fetch stored results
"""
import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import DEV_MODE, MAX_PROMPTS, MIN_PROMPTS, MODELS_CONFIG
from backend.db.database import get_db
from backend.db.models import EvalRun, ModelResult, Prompt, Verdict
from backend.eval.schemas import (
    APIKeys,
    EvalResultsResponse,
    EvalRunRequest,
    EvalRunResponse,
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
    Validates that all referenced image files exist and have valid extensions.
    """
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

        # Validate image files if present
        for row in rows:
            if "image_path" in row and row["image_path"]:
                image_path = row["image_path"]
                # Check if file exists in ZIP
                if image_path not in names:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image file '{image_path}' referenced in manifest.json "
                            f"not found in ZIP. Check the file path and ensure all "
                            f"referenced images are included."
                        ),
                    )
                # Check file extension
                ext = "." + image_path.split(".")[-1].lower() if "." in image_path else ""
                if ext not in VALID_IMAGE_EXTENSIONS:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Image file '{image_path}' has unsupported format '{ext}'. "
                            f"Supported formats: jpg, jpeg, png, webp"
                        ),
                    )

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


# ── Eval run ───────────────────────────────────────────────────────────────


@router.post("/eval/run", response_model=EvalRunResponse)
def start_eval_run(request: EvalRunRequest, db: Session = Depends(get_db)):
    """
    Create an eval run and execute it synchronously in DEV_MODE (mock judge).
    API keys are used in-memory only — never written to the database.
    """
    run_id = str(uuid.uuid4())

    # Use modality from request (client provides it from upload response)
    # Fallback: detect from prompts (for Session 1 backward compat)
    modality = "text"  # will be overridden by client, this is just default

    # Persist run metadata — NOTE: no API key fields in EvalRun model
    run = EvalRun(
        id=run_id,
        created_at=datetime.now(timezone.utc),
        modality=modality,
        rubric_config=request.rubric.model_dump(),
        models_selected=request.models_selected,
        engineer_name=request.engineer_name,
        status="pending",
        custom_label=request.custom_label,
    )
    db.add(run)
    db.commit()

    # Persist prompts (Story 1.4: engineer_name saved here)
    prompt_records: list[Prompt] = []
    for p in request.prompts:
        prompt_rec = Prompt(
            id=str(uuid.uuid4()),
            eval_run_id=run_id,
            prompt_text=p.prompt,
            expected_output=p.expected_output,
            engineer_name=p.engineer_name,  # Story 1.4: engineer tagging saved
        )
        db.add(prompt_rec)
        prompt_records.append(prompt_rec)
    db.commit()

    if DEV_MODE:
        run.status = "running"
        db.commit()

        rubric = request.rubric.model_dump()

        for idx, prompt_rec in enumerate(prompt_records):
            for model_id in request.models_selected:
                # Mock model response (no real API call)
                response_text = get_mock_response(model_id)

                # Mock judge scores (no real API call)
                judge_out = get_mock_judge_scores()
                scores = judge_out["scores"]
                reasoning = judge_out["reasoning"]

                # Calculate cost_efficiency (auto — not judge-scored)
                tokens = {"input": 120, "output": 85}
                cost_usd = calculate_mock_cost(model_id, tokens)
                weighted_score = sum(
                    scores.get(dim, 0) * (rubric.get(dim, 0) / 100)
                    for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]
                )
                cost_efficiency = calculate_cost_efficiency(cost_usd, weighted_score)

                all_scores = {**scores, "cost_efficiency": cost_efficiency}
                hallucination_flagged = scores.get("hallucination", 10) >= 7

                result = ModelResult(
                    id=str(uuid.uuid4()),
                    eval_run_id=run_id,
                    prompt_id=prompt_rec.id,
                    prompt_index=str(idx),
                    model_name=model_id,
                    response_text=response_text,
                    dimension_scores=all_scores,
                    dimension_reasoning=reasoning,
                    hallucination_flagged=hallucination_flagged,
                    hallucination_reason=reasoning.get("hallucination") if hallucination_flagged else None,
                    tokens_used=tokens,
                    cost_usd=cost_usd,
                )
                db.add(result)

        db.commit()

        # Generate minimal verdict (full verdict logic is Session 4)
        winning_model = request.models_selected[0]
        verdict = Verdict(
            id=str(uuid.uuid4()),
            eval_run_id=run_id,
            winning_model=winning_model,
            summary=(
                f"[DEV MODE] Based on mock evaluation scores, {winning_model} is the "
                f"recommended model. Real verdict generation with reasoning is implemented "
                f"in Session 4."
            ),
            score_breakdown={},
            cost_comparison={},
            hallucination_warnings=[],
        )
        db.add(verdict)

        run.status = "complete"
        db.commit()

    return EvalRunResponse(run_id=run_id, status=run.status)


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
