"""
backend/judge/judge.py — LLM-as-Judge scoring (Session 4, Story 2.2)

Scores each model response against the rubric using GPT-5.4 as judge.
Temperature=0 for deterministic scoring.
Retries once on JSON parse failure.
Marks score=null on second failure — run continues.

Hallucination semantics:
  10 = no hallucinations (best quality)
   0 = severe hallucinations (worst)
  flagged=true when score <= 3

DEV_MODE=true: returns mock scores from models.yaml immediately, no API call.
"""
import asyncio
import json
import logging
from typing import Optional

from backend.config import DEV_MODE, DEV_MOCK, MOCK_LATENCY_MS, MODELS_CONFIG, PRICING_CONFIG

logger = logging.getLogger(__name__)

JUDGE_DIMENSIONS = ["accuracy", "hallucination", "instruction_following", "conciseness"]
HALLUCINATION_FLAG_THRESHOLD = 3   # score <= 3 → flagged=true

# Judge model config from models.yaml
_judge_cfg = MODELS_CONFIG["judge"]
JUDGE_MODEL_ID = _judge_cfg["default_model"]                   # "gpt-5-4"
JUDGE_API_MODEL_STRING = PRICING_CONFIG["models"][JUDGE_MODEL_ID]["api_model_string"]  # "gpt-5.4"

# Mock data from models.yaml (DEV_MODE only)
_mock_scores: dict = {k: DEV_MOCK["mock_scores"][k] for k in JUDGE_DIMENSIONS}
_mock_reasoning: dict = {k: str(v).strip() for k, v in DEV_MOCK["mock_reasoning"].items() if k in JUDGE_DIMENSIONS}

# ── Prompt construction ────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are an impartial AI evaluation judge.
Your job is to score AI model responses against a user-defined rubric.
You must always return valid JSON.
You must never return scores without reasoning.
You must quote specific text from the response that influenced each score."""


def _build_dimension_instructions(rubric_config: dict) -> str:
    """Format rubric dimensions with weights and judge instructions from models.yaml."""
    dims_cfg = MODELS_CONFIG["dimensions"]
    lines = []
    for dim in JUDGE_DIMENSIONS:
        info = dims_cfg.get(dim, {})
        weight = rubric_config.get(dim, 0)
        display = info.get("display_name", dim)
        instruction = str(info.get("judge_instruction", "Score 0-10.")).strip()
        lines.append(f"- {display} (weight: {weight}%): {instruction}")
    return "\n".join(lines)


def build_judge_user_prompt(
    prompt_text: str,
    response_text: str,
    rubric_config: dict,
    expected_output: Optional[str],
) -> str:
    """Build the user-turn prompt sent to the judge for a single model response."""
    dim_instructions = _build_dimension_instructions(rubric_config)
    ground_truth = expected_output if expected_output else "Not provided"

    return f"""Evaluate this AI response against the rubric below.

ORIGINAL PROMPT:
{prompt_text}

MODEL RESPONSE:
{response_text}

RUBRIC DIMENSIONS (score each 0-10):
{dim_instructions}

GROUND TRUTH (if provided):
{ground_truth}

Return ONLY this JSON structure, nothing else:
{{
  "scores": {{
    "accuracy": {{
      "score": <0-10>,
      "reasoning": "<one sentence why>",
      "evidence": "<quoted text from response that influenced this score>"
    }},
    "hallucination": {{
      "score": <0-10>,
      "reasoning": "<one sentence why>",
      "evidence": "<quoted excerpt if hallucination detected, or 'None detected'>",
      "flagged": <true|false>
    }},
    "instruction_following": {{
      "score": <0-10>,
      "reasoning": "<one sentence why>",
      "evidence": "<quoted instruction that was followed or missed>"
    }},
    "conciseness": {{
      "score": <0-10>,
      "reasoning": "<one sentence why>",
      "evidence": "<specific observation about length>"
    }}
  }},
  "cost_efficiency": "auto"
}}"""


# ── JSON parsing ───────────────────────────────────────────────────────────


def parse_judge_response(text: str) -> Optional[dict]:
    """
    Parse judge response text into structured dict.
    Returns None on any JSON/schema error.
    """
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if "scores" not in data:
        return None

    scores_raw = data["scores"]
    scores: dict = {}
    reasoning: dict = {}
    evidence: dict = {}

    for dim in JUDGE_DIMENSIONS:
        entry = scores_raw.get(dim, {})
        if not isinstance(entry, dict):
            return None
        score_val = entry.get("score")
        if score_val is None:
            return None
        scores[dim] = float(score_val)
        reasoning[dim] = str(entry.get("reasoning", "")).strip()
        evidence[dim] = str(entry.get("evidence", "")).strip()

    # Validate: every reasoning and evidence must be non-empty
    for dim in JUDGE_DIMENSIONS:
        if not reasoning[dim]:
            return None
        if not evidence[dim]:
            return None

    hallucination_score = scores.get("hallucination", 10.0)
    hallucination_flagged = hallucination_score <= HALLUCINATION_FLAG_THRESHOLD
    hallucination_reason = evidence.get("hallucination") if hallucination_flagged else None

    return {
        "scores": scores,
        "reasoning": reasoning,
        "evidence": evidence,
        "hallucination_flagged": hallucination_flagged,
        "hallucination_reason": hallucination_reason,
    }


# ── Real judge API call ────────────────────────────────────────────────────


async def _call_judge_api(
    user_prompt: str,
    api_key: str,
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
) -> str:
    """Make a single call to the judge model via Responses API. Returns raw response text."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    response = await client.responses.create(
        model=JUDGE_API_MODEL_STRING,
        instructions=system_prompt,
        input=[{"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}],
        temperature=0,
        max_output_tokens=1000,
    )
    return response.output[0].content[0].text


# ── Score single response ──────────────────────────────────────────────────


async def score_response_async(
    prompt_text: str,
    response_text: str,
    rubric_config: dict,
    expected_output: Optional[str],
    api_key: str,
) -> dict:
    """
    Judge a single model response.
    Retries once on JSON parse failure.
    Returns parsed result dict, or a null-score fallback on double failure.
    """
    if DEV_MODE:
        if MOCK_LATENCY_MS > 0:
            await asyncio.sleep(MOCK_LATENCY_MS / 1000 * 0.3)  # judge is faster than model
        hallucination_score = _mock_scores.get("hallucination", 8.5)
        hallucination_flagged = hallucination_score <= HALLUCINATION_FLAG_THRESHOLD
        return {
            "scores": dict(_mock_scores),
            "reasoning": dict(_mock_reasoning),
            "evidence": {dim: f"[MOCK] Evidence for {dim}." for dim in JUDGE_DIMENSIONS},
            "hallucination_flagged": hallucination_flagged,
            "hallucination_reason": _mock_reasoning.get("hallucination") if hallucination_flagged else None,
        }

    user_prompt = build_judge_user_prompt(prompt_text, response_text, rubric_config, expected_output)

    # Attempt 1
    try:
        raw = await _call_judge_api(user_prompt, api_key)
        result = parse_judge_response(raw)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("Judge attempt 1 failed: %s", exc)

    # Retry with explicit JSON instruction
    retry_prompt = user_prompt + "\n\nReturn ONLY valid JSON, no other text."
    try:
        raw = await _call_judge_api(retry_prompt, api_key)
        result = parse_judge_response(raw)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("Judge retry failed: %s", exc)

    # Both attempts failed — return null scores so run continues
    logger.error("Judge returned invalid JSON twice for prompt: %s...", prompt_text[:80])
    null_scores = {dim: None for dim in JUDGE_DIMENSIONS}
    return {
        "scores": null_scores,
        "reasoning": {dim: "" for dim in JUDGE_DIMENSIONS},
        "evidence": {dim: "" for dim in JUDGE_DIMENSIONS},
        "hallucination_flagged": False,
        "hallucination_reason": None,
        "judge_error": True,
    }


# ── Score all results in parallel ─────────────────────────────────────────


async def score_responses_parallel(
    model_results: list[dict],
    rubric_config: dict,
    api_key: str,
) -> list[dict]:
    """
    Judge all model results simultaneously via asyncio.gather().
    Skips results that already have an error (model call failed).
    Returns the same list with judge scores merged in.
    """
    tasks = []
    indices = []  # which results need judging

    for i, r in enumerate(model_results):
        if r.get("error"):
            # Model call failed — skip judging, leave scores empty
            continue
        tasks.append(
            score_response_async(
                r["prompt_text"],
                r["response_text"],
                rubric_config,
                r.get("expected_output"),
                api_key,
            )
        )
        indices.append(i)

    judge_results = await asyncio.gather(*tasks, return_exceptions=True)

    enriched = [dict(r) for r in model_results]
    for idx, judge_out in zip(indices, judge_results):
        if isinstance(judge_out, Exception):
            logger.error("Judge gather error for result %d: %s", idx, judge_out)
            enriched[idx].update(
                {
                    "scores": {dim: None for dim in JUDGE_DIMENSIONS},
                    "reasoning": {dim: "" for dim in JUDGE_DIMENSIONS},
                    "evidence": {dim: "" for dim in JUDGE_DIMENSIONS},
                    "hallucination_flagged": False,
                    "hallucination_reason": None,
                }
            )
        else:
            enriched[idx].update(judge_out)

    return enriched


# ── Ground truth alignment scoring ────────────────────────────────────────

_GT_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating how closely an AI response matches "
    "a given expected output. Return only valid JSON."
)


def _build_gt_user_prompt(expected_output: str, response_text: str) -> str:
    return (
        f"On a scale of 0-10, how closely does this model response match the expected output?\n\n"
        f"Expected:\n{expected_output}\n\n"
        f"Response:\n{response_text}\n\n"
        f'Return ONLY this JSON: {{"alignment_score": <0-10>, "alignment_reasoning": "<one sentence>"}}'
    )


async def score_ground_truth_async(
    expected_output: str,
    response_text: str,
    api_key: str,
) -> dict:
    """
    Score GT alignment for a single response (0-10).
    In DEV_MODE returns a fixed mock score.
    Returns {"ground_truth_score": float, "ground_truth_reasoning": str}.
    """
    if DEV_MODE:
        return {
            "ground_truth_score": 7.5,
            "ground_truth_reasoning": (
                "Response aligns with the expected output in substance and key facts."
            ),
        }

    user_prompt = _build_gt_user_prompt(expected_output, response_text)
    try:
        raw = await _call_judge_api(user_prompt, api_key, system_prompt=_GT_SYSTEM_PROMPT)
        data = json.loads(raw.strip().lstrip("```json").rstrip("```").strip())
        score = min(10.0, max(0.0, float(data["alignment_score"])))
        reasoning = str(data.get("alignment_reasoning", "")).strip()
        return {"ground_truth_score": score, "ground_truth_reasoning": reasoning}
    except Exception as exc:
        logger.warning("GT scoring failed: %s", exc)
        return {"ground_truth_score": None, "ground_truth_reasoning": None}


async def score_ground_truth_parallel(
    scored_results: list[dict],
    api_key: str,
) -> list[dict]:
    """
    Enrich each result that has expected_output with ground_truth_score and
    ground_truth_reasoning.  Results without expected_output get None for both.
    Returns a new list (does not mutate input).
    """
    tasks = []
    indices = []

    for i, r in enumerate(scored_results):
        if r.get("expected_output") and not r.get("error"):
            tasks.append(
                score_ground_truth_async(r["expected_output"], r["response_text"], api_key)
            )
            indices.append(i)

    enriched = [dict(r) for r in scored_results]

    if not tasks:
        for r in enriched:
            r.setdefault("ground_truth_score", None)
            r.setdefault("ground_truth_reasoning", None)
        return enriched

    gt_results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, gt_out in zip(indices, gt_results):
        if isinstance(gt_out, Exception):
            logger.error("GT gather error for result %d: %s", idx, gt_out)
            enriched[idx]["ground_truth_score"] = None
            enriched[idx]["ground_truth_reasoning"] = None
        else:
            enriched[idx].update(gt_out)

    # Fill in None for results that had no expected_output
    for r in enriched:
        r.setdefault("ground_truth_score", None)
        r.setdefault("ground_truth_reasoning", None)

    return enriched

    return enriched
