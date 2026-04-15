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

# Worth-it decision thresholds for cost comparison callout
_WORTH_IT_SCORE_DELTA = 1.0   # premium model must score > 1.0 pts higher
_WORTH_IT_COST_MAX = 0.10     # premium model must cost < $0.10 more per run
_NOT_WORTH_IT_SCORE_DELTA = 0.5  # score delta < 0.5 → not worth it
_NOT_WORTH_IT_COST_MIN = 0.20    # cost delta > $0.20 → not worth it

# Per-dimension insight phrases for variance auto-insight
DIMENSION_INSIGHTS: dict = {
    "accuracy": "this prompt tests factual knowledge where model training data may differ",
    "hallucination": "models vary in tendency to fabricate information on this topic",
    "instruction_following": "models interpreted the instructions differently",
    "conciseness": "models chose very different response lengths for this prompt",
}


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


def generate_cost_comparison_callout(
    model_quality_scores: dict,
    model_total_costs: dict,
    model_ids: list,
    hallucination_flagged_models: set | None = None,
) -> str:
    """
    Auto-generate a one-sentence cost comparison callout.

    Worth-it rule:
      score_delta > 1.0 AND cost_delta < $0.10  → "premium is worth it"
      score_delta < 0.5 OR  cost_delta > $0.20  → "not worth it"
      otherwise                                  → neutral (state the numbers)

    If the cheaper model is hallucination-flagged, the "not worth it" branch
    replaces the pipeline recommendation with a hallucination risk warning.

    Public — tested directly.
    """
    if len(model_ids) < 2:
        return ""

    flagged = hallucination_flagged_models or set()

    sorted_by_cost = sorted(model_ids, key=lambda m: model_total_costs.get(m, 0.0))
    cheapest = sorted_by_cost[0]
    most_expensive = sorted_by_cost[-1]

    if cheapest == most_expensive:
        return ""

    cost_delta = model_total_costs[most_expensive] - model_total_costs[cheapest]
    score_delta = model_quality_scores[most_expensive] - model_quality_scores[cheapest]
    expensive_cost = model_total_costs[most_expensive]
    cheaper_pct = round((cost_delta / expensive_cost) * 100) if expensive_cost > 0 else 0

    if score_delta > _WORTH_IT_SCORE_DELTA and cost_delta < _WORTH_IT_COST_MAX:
        return (
            f"{most_expensive} costs ${cost_delta:.4f} more than {cheapest} "
            f"but scores {round(score_delta, 1)} points higher on quality. "
            f"The premium is worth it for high-stakes annotation work."
        )
    if score_delta < _NOT_WORTH_IT_SCORE_DELTA or cost_delta > _NOT_WORTH_IT_COST_MIN:
        if cheapest in flagged:
            return (
                f"{cheapest} delivers comparable quality at {cheaper_pct}% lower cost "
                f"— however hallucination risk makes it unsuitable for production use "
                f"despite lower cost."
            )
        return (
            f"{cheapest} delivers comparable quality at {cheaper_pct}% lower cost "
            f"— recommended for high-volume pipelines."
        )
    direction = "higher" if score_delta >= 0 else "lower"
    return (
        f"{most_expensive} costs ${cost_delta:.4f} more than {cheapest} "
        f"and scores {round(abs(score_delta), 1)} points {direction}. "
        f"Consider your volume and quality requirements."
    )


def _build_cost_comparison(
    model_total_costs: dict,
    model_quality_scores: dict,
    hallucination_flagged_models: set | None = None,
) -> dict:
    """Build cost_comparison JSON saved to DB and shown in UI. Includes callout."""
    comparison: dict = {}
    for mid, total_cost in model_total_costs.items():
        quality = model_quality_scores.get(mid, 0)
        cpp = round(total_cost / quality, 6) if quality > 0 else 0.0
        comparison[mid] = {
            "total_cost_usd": round(total_cost, 6),
            "cost_per_quality_point": cpp,
        }
    comparison["callout"] = generate_cost_comparison_callout(
        model_quality_scores,
        model_total_costs,
        list(model_total_costs.keys()),
        hallucination_flagged_models=hallucination_flagged_models,
    )
    return comparison


# ── Prompt variance ────────────────────────────────────────────────────────


def calculate_prompt_variance(prompt_results: list, rubric_weights: dict) -> float:
    """
    Variance = max(weighted_score) - min(weighted_score) across models for one prompt.
    Public — tested directly.
    """
    scores = []
    for r in prompt_results:
        ws = calculate_weighted_quality_score(r.get("scores", {}), rubric_weights)
        scores.append(ws)
    if len(scores) < 2:
        return 0.0
    return round(max(scores) - min(scores), 4)


def generate_variance_insight(prompt_results: list) -> str:
    """
    Find the dimension with the largest per-prompt score delta across models.
    Returns a one-sentence insight mentioning that dimension.
    Public — tested directly.
    """
    if len(prompt_results) < 2:
        return ""

    max_delta = 0.0
    max_dim: Optional[str] = None
    for dim in JUDGE_DIMENSIONS:
        vals = [
            float(r["scores"][dim])
            for r in prompt_results
            if r.get("scores", {}).get(dim) is not None
        ]
        if len(vals) < 2:
            continue
        delta = max(vals) - min(vals)
        if delta > max_delta:
            max_delta = delta
            max_dim = dim

    if max_dim is None:
        return ""

    phrase = DIMENSION_INSIGHTS.get(max_dim, "models showed different performance")
    dim_display = max_dim.replace("_", " ")
    return (
        f"Models differed most on {dim_display} (\u03b4={max_delta:.1f}) \u2014 {phrase}."
    )


def rank_prompts_by_variance(
    scored_results: list,
    rubric_config: dict,
) -> list:
    """
    Returns list of (prompt_id, variance_score) sorted by variance descending.
    Public — tested directly.
    """
    by_prompt: dict = defaultdict(list)
    for r in scored_results:
        if not r.get("error"):
            by_prompt[r["prompt_id"]].append(r)

    variances = [
        (pid, calculate_prompt_variance(results, rubric_config))
        for pid, results in by_prompt.items()
    ]
    return sorted(variances, key=lambda x: x[1], reverse=True)


def get_high_variance_prompt_ids(
    ranked_prompts: list,
    top_n: int = 3,
) -> set:
    """
    Returns the set of top_n prompt_ids by variance.
    Public — tested directly.
    """
    return {pid for pid, _ in ranked_prompts[:top_n]}


def save_variance_scores(
    run_id: str,
    scored_results: list,
    rubric_config: dict,
    db,
) -> None:
    """
    Calculate variance per prompt and persist to model_results.variance_score.
    Called after generate_verdict() while the same DB session is still open.
    """
    from backend.db.models import ModelResult  # deferred to avoid circular import

    ranked = rank_prompts_by_variance(scored_results, rubric_config)
    for prompt_id, variance in ranked:
        db.query(ModelResult).filter(
            ModelResult.eval_run_id == run_id,
            ModelResult.prompt_id == prompt_id,
        ).update({"variance_score": variance})
    db.commit()


# ── Verdict text generation ────────────────────────────────────────────────


def build_gt_alignment_summary(scored_results: list) -> str:
    """
    Build one-line GT alignment sentence for the verdict.
    Returns empty string if no results carry a ground_truth_score.
    Public — tested directly.
    """
    gt_sum: dict = defaultdict(float)
    gt_cnt: dict = defaultdict(int)
    for r in scored_results:
        gts = r.get("ground_truth_score")
        if gts is not None:
            mid = r["model_id"]
            gt_sum[mid] += float(gts)
            gt_cnt[mid] += 1
    if not gt_cnt:
        return ""
    gt_avgs = {mid: gt_sum[mid] / gt_cnt[mid] for mid in gt_cnt}
    best = max(gt_avgs, key=lambda m: gt_avgs[m])
    avg = round(gt_avgs[best], 1)
    return (
        f"Against ground truth: {best} aligned most closely with expected outputs "
        f"(avg alignment: {avg}/10)"
    )


def build_verdict_text(
    winning_model: str,
    final_score: float,
    other_models: list[str],
    top_dimensions: list[str],
    cost_insight: str,
    hallucination_warnings: list[str],
    gt_summary: str = "",
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

    if gt_summary:
        lines.append("")
        lines.append(gt_summary)

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

    gt_summary = build_gt_alignment_summary(scored_results)

    summary = build_verdict_text(
        winning_model=winning_model,
        final_score=winning_score,
        other_models=other_models,
        top_dimensions=top_dim_names,
        cost_insight=cost_insight,
        hallucination_warnings=hallucination_warnings,
        gt_summary=gt_summary,
    )

    score_breakdown = _build_score_breakdown(
        model_quality_scores, model_cost_eff_scores, model_final_scores, rubric_config, model_dim_averages
    )
    cost_comparison = _build_cost_comparison(
        model_total_costs, model_quality_scores, hallucination_flagged_models=disqualified
    )

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
