"""
backend/export/json_exporter.py — Story 3.2

Builds the canonical VerdictAI JSON export schema.
Returned as a Python dict; FastAPI serialises it to JSON.

Schema version: 1.0.0
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone


VERDICTAI_VERSION = "1.0.0"
DIMS = ["accuracy", "hallucination", "instruction_following", "conciseness"]


def _iso(dt: datetime | None) -> str | None:
    """Return ISO-8601 string (UTC) or None."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.isoformat()


def _detect_preset(rubric: dict) -> str:
    """Return matching preset name or 'custom'."""
    try:
        import os, yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "models.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for preset in cfg.get("rubric_presets", []):
            pw = preset.get("weights", {})
            if all(rubric.get(k) == pw.get(k) for k in DIMS + ["cost_efficiency"]):
                return preset.get("name", "custom")
    except Exception:
        pass
    return "custom"


def generate_json_report(run_id: str, db) -> dict:
    """
    Build and return the full JSON export for a completed eval run.

    Structure:
      verdictai_version, exported_at, run, verdict, models, prompts
    """
    from backend.db.models import EvalRun, ModelResult, Prompt, Verdict

    # ── Fetch from DB ──────────────────────────────────────────────────────
    run         = db.query(EvalRun).filter(EvalRun.id == run_id).first()
    if not run:
        raise ValueError(f"Run '{run_id}' not found.")

    verdict_row = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
    results     = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
    prompts     = db.query(Prompt).filter(Prompt.eval_run_id == run_id).all()

    # ── Derived lookups ─────────────────────────────────────────────────────
    prompt_by_id  = {p.id: p for p in prompts}
    prompt_order  = {p.id: i for i, p in enumerate(prompts)}
    models        = sorted({r.model_name for r in results})
    rubric_config = run.rubric_config or {}

    # ── Aggregate per-model stats ──────────────────────────────────────────
    cost_by_model:        dict = defaultdict(float)
    judge_cost_by_model:  dict = defaultdict(float)
    gt_cost_by_model:     dict = defaultdict(float)
    tin_by_model:         dict = defaultdict(int)
    tout_by_model:        dict = defaultdict(int)
    eval_calls_by_model:  dict = defaultdict(int)
    judge_calls_by_model: dict = defaultdict(int)
    gt_calls_by_model:    dict = defaultdict(int)
    dim_sum:              dict = defaultdict(lambda: defaultdict(float))
    dim_cnt:              dict = defaultdict(lambda: defaultdict(int))

    for r in results:
        m = r.model_name
        cost_by_model[m]        += r.cost_usd or 0.0
        judge_cost_by_model[m]  += r.judge_cost_usd or 0.0
        gt_cost_by_model[m]     += r.gt_cost_usd or 0.0
        tin_by_model[m]         += r.tokens_in or 0
        tout_by_model[m]        += r.tokens_out or 0
        eval_calls_by_model[m]  += r.eval_api_calls or 0
        judge_calls_by_model[m] += r.judge_api_calls or 0
        gt_calls_by_model[m]    += r.gt_api_calls or 0
        for d in DIMS:
            v = (r.dimension_scores or {}).get(d)
            if v is not None:
                dim_sum[m][d] += float(v)
                dim_cnt[m][d] += 1

    score_bd  = (verdict_row.score_breakdown or {}) if verdict_row else {}
    cost_comp = (verdict_row.cost_comparison or {}) if verdict_row else {}

    # ── verdict section ────────────────────────────────────────────────────
    winning_model = verdict_row.winning_model if verdict_row else (models[0] if models else None)
    overall_score = None
    if winning_model and winning_model in score_bd:
        overall_score = score_bd[winning_model].get("final_score")

    # GT alignment summary
    gt_sum: dict = defaultdict(float)
    gt_cnt: dict = defaultdict(int)
    for r in results:
        if r.ground_truth_score is not None:
            gt_sum[r.model_name] += float(r.ground_truth_score)
            gt_cnt[r.model_name] += 1

    gt_alignment_summary = None
    if gt_cnt:
        best_gt = max(gt_cnt, key=lambda m: gt_sum[m] / gt_cnt[m])
        avg_gt  = round(gt_sum[best_gt] / gt_cnt[best_gt], 2)
        gt_alignment_summary = {"best_model": best_gt, "avg_score": avg_gt}

    # ── per-prompt variance map ────────────────────────────────────────────
    variance_by_prompt: dict = {}
    for r in results:
        if r.variance_score is not None and r.prompt_id not in variance_by_prompt:
            variance_by_prompt[r.prompt_id] = float(r.variance_score)

    HIGH_VAR_THRESHOLD = 0.5

    # ── models section ─────────────────────────────────────────────────────
    models_section: dict = {}
    for m in models:
        total_c = cost_by_model[m]
        tin     = tin_by_model[m]
        tout    = tout_by_model[m]
        total_t = tin + tout
        cost_1k = round(total_c / total_t * 1000, 6) if total_t > 0 else None
        final   = score_bd.get(m, {}).get("final_score")
        q_score = final or 1.0
        cpp     = round(total_c / q_score, 6) if q_score > 0 else None

        avg_scores: dict = {}
        for d in DIMS:
            avg_scores[d] = round(dim_sum[m][d] / dim_cnt[m][d], 4) if dim_cnt[m][d] > 0 else None
        ce_score = score_bd.get(m, {}).get("cost_efficiency_score")
        avg_scores["cost_efficiency"] = round(float(ce_score), 4) if ce_score is not None else None
        avg_scores["total"]           = round(float(final), 4) if final is not None else None

        j_cost = judge_cost_by_model[m]
        g_cost = gt_cost_by_model[m]

        models_section[m] = {
            "avg_scores": avg_scores,
            "cost": {
                "eval_cost_usd":          round(total_c, 6),
                "judge_cost_usd":         round(j_cost, 6),
                "gt_cost_usd":            round(g_cost, 6),
                "total_usd":              round(total_c + j_cost + g_cost, 6),
                "tokens_in":              tin,
                "tokens_out":             tout,
                "cost_per_1k_tokens":     cost_1k,
                "cost_per_quality_point": cpp,
            },
            "api_calls": {
                "eval_calls":  eval_calls_by_model[m],
                "judge_calls": judge_calls_by_model[m],
                "gt_calls":    gt_calls_by_model[m],
                "total_calls": eval_calls_by_model[m] + judge_calls_by_model[m] + gt_calls_by_model[m],
            },
        }

    # ── prompts section ────────────────────────────────────────────────────
    # Group results by prompt
    by_prompt: dict = defaultdict(list)
    for r in results:
        by_prompt[r.prompt_id].append(r)

    prompts_section = []
    for p in prompts:
        p_results    = by_prompt.get(p.id, [])
        p_idx        = prompt_order.get(p.id, 0)
        p_variance   = variance_by_prompt.get(p.id)
        p_high_var   = (p_variance is not None and p_variance >= HIGH_VAR_THRESHOLD)

        responses: dict = {}
        for r in p_results:
            ds = r.dimension_scores or {}
            dr = r.dimension_reasoning or {}
            de = r.evidence_data or {}
            scores_dict: dict = {}
            for d in DIMS:
                scores_dict[d] = {
                    "score":     ds.get(d),
                    "reasoning": dr.get(d),
                    "evidence":  de.get(d),
                }

            responses[r.model_name] = {
                "response_text":         r.response_text,
                "scores":                scores_dict,
                "ground_truth_score":    r.ground_truth_score,
                "ground_truth_reasoning":r.ground_truth_reasoning,
                "rouge_1_score":         r.rouge_1_score,
                "rouge_l_score":         r.rouge_l_score,
                "hallucination_flagged": r.hallucination_flagged,
                "tokens_in":             r.tokens_in,
                "tokens_out":            r.tokens_out,
                "cost_usd":              r.cost_usd,
                "judge_tokens_in":       r.judge_tokens_in,
                "judge_tokens_out":      r.judge_tokens_out,
                "judge_cost_usd":        r.judge_cost_usd,
                "gt_tokens_in":          r.gt_tokens_in,
                "gt_tokens_out":         r.gt_tokens_out,
                "gt_cost_usd":           r.gt_cost_usd,
                "eval_api_calls":        r.eval_api_calls,
                "judge_api_calls":       r.judge_api_calls,
                "gt_api_calls":          r.gt_api_calls,
            }

        prompts_section.append({
            "index":           p_idx,
            "prompt_text":     p.prompt_text,
            "engineer_name":   p.engineer_name,
            "expected_output": p.expected_output,
            "variance_score":  p_variance,
            "high_variance":   p_high_var,
            "responses":       responses,
        })

    # Sort by prompt index for deterministic order
    prompts_section.sort(key=lambda x: x["index"])

    # ── API calls summary ──────────────────────────────────────────────────
    total_eval_calls_all  = sum(eval_calls_by_model.values())
    total_judge_calls_all = sum(judge_calls_by_model.values())
    total_gt_calls_all    = sum(gt_calls_by_model.values())

    # ── Assemble final schema ──────────────────────────────────────────────
    return {
        "verdictai_version": VERDICTAI_VERSION,
        "exported_at":       _iso(datetime.now(timezone.utc)),
        "api_calls_summary": {
            "eval_calls":  total_eval_calls_all,
            "judge_calls": total_judge_calls_all,
            "gt_calls":    total_gt_calls_all,
            "total_calls": total_eval_calls_all + total_judge_calls_all + total_gt_calls_all,
            "total_eval_cost_usd":  round(sum(cost_by_model.values()), 6),
            "total_judge_cost_usd": round(sum(judge_cost_by_model.values()), 6),
            "total_gt_cost_usd":    round(sum(gt_cost_by_model.values()), 6),
        },
        "run": {
            "id":               run.id,
            "label":            run.custom_label,
            "created_at":       _iso(run.created_at),
            "completed_at":     _iso(run.completed_at),
            "modality":         run.modality,
            "models_selected":  run.models_selected or [],
            "rubric_config": {
                "preset":  _detect_preset(rubric_config),
                "weights": {
                    "accuracy":               rubric_config.get("accuracy", 0),
                    "hallucination":          rubric_config.get("hallucination", 0),
                    "instruction_following":  rubric_config.get("instruction_following", 0),
                    "conciseness":            rubric_config.get("conciseness", 0),
                    "cost_efficiency":        rubric_config.get("cost_efficiency", 0),
                },
            },
            "engineer_names": run.engineer_names or [],
        },
        "verdict": {
            "winning_model":          winning_model,
            "overall_score":          round(float(overall_score), 4) if overall_score is not None else None,
            "summary":                verdict_row.summary if verdict_row else None,
            "hallucination_warnings": verdict_row.hallucination_warnings if verdict_row else [],
            "gt_alignment_summary":   gt_alignment_summary,
        },
        "models":  models_section,
        "prompts": prompts_section,
    }
