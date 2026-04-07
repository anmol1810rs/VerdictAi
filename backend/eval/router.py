"""
eval/router.py — Session 1 endpoints.

Implements:
  POST /upload          — parse CSV / JSONL / ZIP(image+text)
  POST /rubric/validate — validate rubric weights
  POST /keys/validate   — validate API key format
  POST /eval/run        — create run, execute mock judge (DEV_MODE), persist to DB
  GET  /eval/{run_id}/results — fetch stored results
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

from backend.config import DEV_MODE, MAX_PROMPTS, MIN_PROMPTS
from backend.db.database import get_db
from backend.db.models import EvalRun, ModelResult, Prompt, Verdict
from backend.eval.schemas import (
    APIKeys,
    EvalResultsResponse,
    EvalRunRequest,
    EvalRunResponse,
    KeyValidationResponse,
    ModelResultOut,
    PromptInput,
    RubricWeights,
    UploadResponse,
)
from backend.judge.mock_judge import (
    calculate_cost_efficiency,
    calculate_mock_cost,
    get_mock_judge_scores,
    get_mock_response,
)

router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────


def _parse_csv(content: bytes) -> list[dict]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader]


def _parse_jsonl(content: bytes) -> list[dict]:
    lines = content.decode("utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _parse_zip(content: bytes) -> tuple[list[dict], str]:
    """Parse a ZIP containing manifest.json + optional images folder."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        names = z.namelist()
        manifest_names = [n for n in names if n.endswith("manifest.json")]
        if not manifest_names:
            raise HTTPException(
                status_code=422,
                detail=(
                    "ZIP must contain a manifest.json file. "
                    "See the upload template for the expected format."
                ),
            )
        manifest_data = json.loads(z.read(manifest_names[0]))
        rows = manifest_data if isinstance(manifest_data, list) else manifest_data.get("prompts", [])
    return rows, "image_text"


def _rows_to_prompts(rows: list[dict]) -> list[PromptInput]:
    """Convert parsed rows to PromptInput objects with validation."""
    prompts = []
    for i, row in enumerate(rows, start=1):
        if "prompt" not in row or not str(row.get("prompt", "")).strip():
            raise HTTPException(
                status_code=422,
                detail=f"Row {i} is missing a non-empty 'prompt' column.",
            )
        prompts.append(
            PromptInput(
                prompt=str(row["prompt"]).strip(),
                expected_output=row.get("expected_output") or None,
                engineer_name=row.get("engineer_name") or None,
            )
        )
    return prompts


def _validate_prompt_count(count: int) -> None:
    if count == 0:
        raise HTTPException(
            status_code=422,
            detail="Dataset is empty. Upload at least 5 prompts.",
        )
    if count < MIN_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail=f"Dataset must contain at least {MIN_PROMPTS} prompts, got {count}.",
        )
    if count > MAX_PROMPTS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Dataset exceeds the MVP maximum of {MAX_PROMPTS} prompts (got {count}). "
                f"Reduce to {MAX_PROMPTS} or fewer."
            ),
        )


# ── Story 1.1 — Upload ─────────────────────────────────────────────────────


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    content = await file.read()

    if filename.endswith(".csv"):
        rows = _parse_csv(content)
        modality = "text"
    elif filename.endswith(".jsonl"):
        rows = _parse_jsonl(content)
        modality = "text"
    elif filename.endswith(".zip"):
        rows, modality = _parse_zip(content)
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file type. Upload a .csv, .jsonl, or .zip file.",
        )

    _validate_prompt_count(len(rows))
    prompts = _rows_to_prompts(rows)

    return UploadResponse(
        prompt_count=len(prompts),
        has_ground_truth=any(p.expected_output for p in prompts),
        has_engineer_names=any(p.engineer_name for p in prompts),
        modality=modality,
        prompts=prompts,
    )


# ── Story 1.2 — API keys ───────────────────────────────────────────────────


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
    Real async multi-model execution is implemented in Session 4.
    """
    run_id = str(uuid.uuid4())

    # Detect modality from prompt content
    modality = "image_text" if any(p.engineer_name for p in request.prompts) else "text"

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

    # Persist prompts
    prompt_records: list[Prompt] = []
    for p in request.prompts:
        prompt_rec = Prompt(
            id=str(uuid.uuid4()),
            eval_run_id=run_id,
            prompt_text=p.prompt,
            expected_output=p.expected_output,
            engineer_name=p.engineer_name,
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
