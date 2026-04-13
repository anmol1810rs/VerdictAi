"""
backend/verdict/verdict.py — Verdict generation (Session 4, Story 2.1)

After all model responses are scored by the judge, this module:
  1. Calculates weighted quality score per model (excludes cost_efficiency)
  2. Calculates cost_efficiency scores (normalized: lowest cost = 10, highest = 0)
  3. Applies hallucination penalty: model with >30% flagged prompts cannot win
  4. Selects winner by highest final score
  5. Generates plain-language verdict text
  6. Saves Verdict row to DB

All calculation functions are public and tested directly.
"""
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Optional

JUDGE_DIMENSIONS = ["accuracy", "hallucination", "instruction_following", "conciseness"]
HALLUCINATION_DISQUALIFY_PCT = 0.30  # >30% prompts flagged → cannot win


# ── Score calculations ─────────────────────────────────────────────────────


def calculate_weighted_quality_score(scores: dict, rubric_weights: dict) -> float:
    """
    Weighted average of the four judge dimensions (excludes cost_efficiency).
    Each dimension contributes: score × (weight / 100).
    Result is on a 0-10 scale (assuming scores are 0-10 and all weights sum
    to 100 when cost_efficiency is included — quality dims will sum < 100).

    Skips None scores (judge failed) treating them as 0.
    """
    total = 0.0
    for dim in JUDGE_DIMENSIONS:
        score = scores.get(dim)
        weight = rubric_weights.get(dim, 0)
        if score is not None:
            total += float(score) * (weight / 100)
    return round(total, 4)


def normalize_cost_efficiency(model_cpp: dict) -> dict:
    """
    Normalize cost-per-quality-point across models.
    Lowest cpp → score 10 (best), highest cpp → score 0 (worst).
    If all models have equal cpp: all get 5.0.

    Args:
        model_cpp: {model_id: cost_per_quality_point}
    Returns:
        {model_id: normalized_score_0_to_10}
    """
    if not model_cpp:
        return {}

    min_cpp = min(model_cpp.values())
    max_cpp = max(model_cpp.values())

    if max_cpp == min_cpp:
        return {m: 5.0 for m in model_cpp}

    return {
        m: round(10.0 * (max_cpp - cpp) / (max_cpp - min_cpp), 4)
        for m, cpp in model_cpp.items()
    }


def detect_hallucination_disqualified(
    scored_results: list[dict],
) -> set:
    """
    Returns set of model_ids that are disqualified due to hallucination.
    Disqualified if > 30% of their prompts have hallucination_flagged=True.
    """
    model_total: dict = defaultdict(int)
    model_flagged: dict = defaultdict(int)

    for r in scored_results:
        if r.get("error"):
            continue
        mid = r["model_id"]
        model_total[mid] += 1
        if r.get("hallucination_flagged", False):
            model_flagged[mid] += 1

    disqualified = set()
    for mid, total in model_total.items():
        if total > 0 and model_flagged[mid] / total > HALLUCINATION_DISQUALIFY_PCT:
            disqualified.add(mid)
    return disqualified


def _build_score_breakdown(
    model_quality_scores: dict,
    model_cost_eff_scores: dict,
    model_final_scores: dict,
    rubric_weights: dict,
    model_dim_averages: dict,
) -> dict:
    """Build the score_breakdown JSON saved to DB and shown in UI."""
    breakdown = {}
    for mid in model_final_scores:
        breakdown[mid] = {
            "final_score": round(model_final_scores[mid], 3),
            "quality_score": round(model_quality_scores[mid], 3),
            "cost_efficiency_score": round(model_cost_eff_scores.get(mid, 0.0), 3),
            "dimensions": {
                dim: round(model_dim_averages[mid].get(dim, 0.0), 3)
                for dim in JUDGE_DIMENSIONS
            },
        }
    return breakdown


def _build_cost_comparison(model_total_costs: dict, model_quality_scores: dict) -> dict:
    """Build cost_comparison JSON saved to DB and shown in UI."""
    comparison = {}
    for mid, total_cost in model_total_costs.items():
        quality = model_quality_scores.get(mid, 0)
        cpp = round(total_cost / quality, 6) if quality > 0 else 0.0
        comparison[mid] = {
            "total_cost_usd": round(total_cost, 6),
            "cost_per_quality_point": cpp,
        }
    return comparison


# ── Verdict text generation ────────────────────────────────────────────────


def build_verdict_text(
    winning_model: str,
    final_score: float,
    other_models: list[str],
    top_dimensions: list[str],
    cost_insight: str,
    hallucination_warnings: list[str],
) -> str:
    """
    Generate plain-language verdict text per PRD template.
    Public — tested directly.
    """
    others_str = ", ".join(other_models) if other_models else "all other models"
    top_dims_str = ", ".join(top_dimensions) if top_dimensions else "overall quality"

    lines = [
        f"{winning_model} is recommended for your use case.",
        "",
        f"It scored {round(final_score, 2)}/10 overall, outperforming "
        f"{others_str} on {top_dims_str}.",
    ]

    if cost_insight:
        lines.append("")
        lines.append(cost_insight)

    for warning in hallucination_warnings:
        lines.append("")
        lines.append(f"⚠️ {warning}")

    return "\n".join(lines)


# ── Main verdict entry point ───────────────────────────────────────────────


def generate_verdict(
    run_id: str,
    scored_results: list[dict],
    rubric_config: dict,
    db,  # SQLAlchemy Session
) -> None:
    """
    Generate and save a Verdict row for the given eval run.
    Called after all ModelResults have been saved.

    scored_results: output of score_responses_parallel — each item has:
        model_id, prompt_id, prompt_index, response_text,
        tokens_in, tokens_out, cost_usd,
        scores, reasoning, hallucination_flagged, hallucination_reason
    """
    from backend.db.models import Verdict  # deferred to avoid circular import at module load

    if not scored_results:
        return

    # Collect unique model IDs
    model_ids = list({r["model_id"] for r in scored_results if not r.get("error")})
    if not model_ids:
        return

    # ── Per-model aggregation ──────────────────────────────────────────────

    model_total_costs: dict = defaultdict(float)
    model_dim_sum: dict = {mid: defaultdict(float) for mid in model_ids}
    model_dim_count: dict = {mid: defaultdict(int) for mid in model_ids}

    for r in scored_results:
        if r.get("error"):
            continue
        mid = r["model_id"]
        model_total_costs[mid] += r.get("cost_usd", 0.0)
        scores = r.get("scores", {})
        for dim in JUDGE_DIMENSIONS:
            v = scores.get(dim)
            if v is not None:
                model_dim_sum[mid][dim] += float(v)
                model_dim_count[mid][dim] += 1

    # Average dimension scores per model
    model_dim_averages: dict = {}
    for mid in model_ids:
        model_dim_averages[mid] = {
            dim: (
                model_dim_sum[mid][dim] / model_dim_count[mid][dim]
                if model_dim_count[mid][dim] > 0
                else 0.0
            )
            for dim in JUDGE_DIMENSIONS
        }

    # ── Weighted quality scores ────────────────────────────────────────────

    model_quality_scores: dict = {
        mid: calculate_weighted_quality_score(model_dim_averages[mid], rubric_config)
        for mid in model_ids
    }

    # ── Cost efficiency ────────────────────────────────────────────────────

    model_cpp: dict = {
        mid: (
            model_total_costs[mid] / model_quality_scores[mid]
            if model_quality_scores[mid] > 0
            else float("inf")
        )
        for mid in model_ids
    }
    # Replace inf with a large finite number for normalization
    finite_max = max((v for v in model_cpp.values() if v != float("inf")), default=1.0)
    model_cpp_finite = {m: (v if v != float("inf") else finite_max * 2) for m, v in model_cpp.items()}
    model_cost_eff_scores = normalize_cost_efficiency(model_cpp_finite)

    cost_eff_weight = rubric_config.get("cost_efficiency", 0)

    # ── Final scores = quality + cost_efficiency component ─────────────────

    model_final_scores: dict = {
        mid: model_quality_scores[mid] + model_cost_eff_scores.get(mid, 0.0) * (cost_eff_weight / 100)
        for mid in model_ids
    }

    # ── Hallucination disqualification ─────────────────────────────────────

    disqualified = detect_hallucination_disqualified(scored_results)
    eligible_models = [mid for mid in model_ids if mid not in disqualified]

    if not eligible_models:
        # All disqualified — pick highest final score anyway with warning
        eligible_models = model_ids

    # ── Winner selection ───────────────────────────────────────────────────

    winning_model = max(eligible_models, key=lambda m: model_final_scores[m])
    winning_score = model_final_scores[winning_model]
    other_models = [m for m in model_ids if m != winning_model]

    # ── Top dimensions for winner (top 2 by dim score) ────────────────────

    dim_scores_winner = model_dim_averages[winning_model]
    top_dims = sorted(JUDGE_DIMENSIONS, key=lambda d: dim_scores_winner.get(d, 0), reverse=True)[:2]
    top_dim_names = [d.replace("_", " ").title() for d in top_dims]

    # ── Cost insight ───────────────────────────────────────────────────────

    winner_cost = model_total_costs.get(winning_model, 0.0)
    cost_insight = ""
    if other_models:
        cheapest_other = min(other_models, key=lambda m: model_total_costs.get(m, 0.0))
        other_cost = model_total_costs.get(cheapest_other, 0.0)
        if winner_cost <= other_cost:
            cost_insight = (
                f"At ${winner_cost:.6f} per eval run, it is also the most cost-efficient option."
            )
        else:
            diff = winner_cost - other_cost
            quality_diff = model_quality_scores[winning_model] - model_quality_scores.get(cheapest_other, 0.0)
            cost_insight = (
                f"It costs ${diff:.6f} more than {cheapest_other} "
                f"but scores {round(quality_diff, 2)} points higher on quality."
            )

    # ── Hallucination warnings ─────────────────────────────────────────────

    hallucination_warnings: list[str] = []
    model_total_count: dict = defaultdict(int)
    model_flagged_count: dict = defaultdict(int)
    for r in scored_results:
        if r.get("error"):
            continue
        mid = r["model_id"]
        model_total_count[mid] += 1
        if r.get("hallucination_flagged", False):
            model_flagged_count[mid] += 1

    for mid in disqualified:
        n_flagged = model_flagged_count[mid]
        n_total = model_total_count[mid]
        hallucination_warnings.append(
            f"{mid} was flagged for hallucinations on "
            f"{n_flagged} of {n_total} prompts and is not recommended for production use."
        )

    # ── Build verdict text ─────────────────────────────────────────────────

    summary = build_verdict_text(
        winning_model=winning_model,
        final_score=winning_score,
        other_models=other_models,
        top_dimensions=top_dim_names,
        cost_insight=cost_insight,
        hallucination_warnings=hallucination_warnings,
    )

    score_breakdown = _build_score_breakdown(
        model_quality_scores, model_cost_eff_scores, model_final_scores, rubric_config, model_dim_averages
    )
    cost_comparison = _build_cost_comparison(model_total_costs, model_quality_scores)

    # ── Save to DB ─────────────────────────────────────────────────────────

    verdict = Verdict(
        id=str(uuid.uuid4()),
        eval_run_id=run_id,
        winning_model=winning_model,
        summary=summary,
        score_breakdown=score_breakdown,
        cost_comparison=cost_comparison,
        hallucination_warnings=hallucination_warnings,
        created_at=datetime.utcnow(),
    )
    db.add(verdict)
    db.commit()
