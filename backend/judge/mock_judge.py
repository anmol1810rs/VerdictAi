"""
Mock judge — used when DEV_MODE=true.

Returns realistic hardcoded scores from models.yaml dev_mode config.
Zero API cost, no network calls. Simulates latency for realistic UX testing.
"""
import time

from backend.config import DEV_MOCK, MOCK_LATENCY_MS

# Pull mock values directly from models.yaml
_raw_scores: dict = DEV_MOCK["mock_scores"]
_reasoning: dict = DEV_MOCK["mock_reasoning"]
_mock_responses: dict = DEV_MOCK["mock_models"]

# Dimensions scored by judge (cost_efficiency is auto-calculated, not judge-scored)
JUDGE_DIMENSIONS = ["accuracy", "hallucination", "instruction_following", "conciseness"]

MOCK_SCORES: dict = {k: _raw_scores[k] for k in JUDGE_DIMENSIONS}
MOCK_REASONING: dict = {k: str(v).strip() for k, v in _reasoning.items() if k in JUDGE_DIMENSIONS}

# Mock token usage per call (realistic approximation for cost calculation)
MOCK_TOKENS = {"input": 120, "output": 85}


def get_mock_response(model_id: str) -> str:
    """Return the mock response text for a given model ID."""
    if MOCK_LATENCY_MS > 0:
        time.sleep(MOCK_LATENCY_MS / 1000)
    return _mock_responses.get(model_id, f"[MOCK] Response from {model_id} for development testing.")


def get_mock_judge_scores() -> dict:
    """Return mock dimension scores and reasoning from models.yaml."""
    return {
        "scores": dict(MOCK_SCORES),
        "reasoning": dict(MOCK_REASONING),
    }


def calculate_mock_cost(model_id: str, tokens: dict) -> float:
    """Return a plausible mock cost in USD (not real pricing — DEV_MODE only)."""
    return round((tokens["input"] + tokens["output"]) * 0.000002, 6)


def get_mock_gt_score(expected_output: str | None) -> tuple:
    """
    Return (gt_score, gt_reasoning) when expected_output is present, else (None, None).
    Used in DEV_MODE to avoid real judge API calls for GT scoring.
    """
    if not expected_output:
        return None, None
    return 7.5, "Response aligns with the expected output in substance and key facts."


def calculate_cost_efficiency(cost_usd: float, weighted_score: float) -> float:
    """cost_efficiency = cost / weighted_score. Lower is better."""
    if weighted_score == 0:
        return 0.0
    return round(cost_usd / weighted_score, 6)
