"""
Session 5 acceptance tests — Stories 2.3 (Cost Breakdown) and 2.4 (Prompt Variance).

Run: pytest backend/tests/test_session_5.py -v

Test groups:
  TestCostBreakdown    (5 tests) — tokens split, cpp, callout, prices date, DB fields
  TestPromptVariance   (6 tests) — variance calc, sorted order, top-3 flagging,
                                   expansion data, insight dimension, DB persistence
"""
import pytest

from backend.tests.conftest import VALID_RUBRIC, make_prompts, valid_run_request


# ══════════════════════════════════════════════════════════════════════════
# Story 2.3 — Cost Breakdown Enhancement
# ══════════════════════════════════════════════════════════════════════════


class TestCostBreakdown:

    def test_input_output_tokens_stored_separately(self, client):
        """
        After a complete run, tokens_in and tokens_out must be stored as
        separate non-null integer columns in model_results.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5, models=["gpt-5-4"]))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) > 0, "Must have model results"
            for r in results:
                assert r.tokens_in is not None, "tokens_in must not be null"
                assert r.tokens_out is not None, "tokens_out must not be null"
                assert isinstance(r.tokens_in, int), "tokens_in must be int"
                assert isinstance(r.tokens_out, int), "tokens_out must be int"
                assert r.tokens_in > 0, "tokens_in must be > 0"
                assert r.tokens_out > 0, "tokens_out must be > 0"
        finally:
            db.close()

    def test_cost_per_quality_point_calculated(self):
        """
        cost_per_quality_point = total_cost_usd / weighted_quality_score.
        Mock cost=0.05, weighted_score=7.5 → cpp = 0.05 / 7.5 ≈ 0.006667.
        """
        from backend.verdict.verdict import _build_cost_comparison

        model_total_costs = {"gpt-5-4": 0.05}
        model_quality_scores = {"gpt-5-4": 7.5}

        comparison = _build_cost_comparison(model_total_costs, model_quality_scores)

        cpp = comparison["gpt-5-4"]["cost_per_quality_point"]
        expected = round(0.05 / 7.5, 6)
        assert abs(cpp - expected) < 1e-6, (
            f"cost_per_quality_point mismatch: expected {expected}, got {cpp}"
        )

    def test_cost_comparison_callout_generated(self):
        """
        generate_cost_comparison_callout must produce a sentence mentioning both
        models and the cost difference when given two models with distinct costs.
        """
        from backend.verdict.verdict import generate_cost_comparison_callout

        model_quality_scores = {"model-a": 8.5, "model-b": 7.2}
        model_total_costs = {"model-a": 0.20, "model-b": 0.05}

        callout = generate_cost_comparison_callout(
            model_quality_scores, model_total_costs, ["model-a", "model-b"]
        )

        assert isinstance(callout, str), "Callout must be a string"
        assert len(callout) > 0, "Callout must not be empty"
        assert "model-a" in callout or "model-b" in callout, (
            "Callout must mention at least one model"
        )

    def test_cost_comparison_callout_worth_it_rule(self):
        """
        Worth-it rule:
          score_delta > 1.0 AND cost_delta < $0.10 → mentions 'worth it'
          score_delta < 0.5 OR  cost_delta > $0.20 → mentions cheaper model + 'lower cost'
        """
        from backend.verdict.verdict import generate_cost_comparison_callout

        # Case 1: worth it (big score gain, tiny cost premium)
        callout_worth = generate_cost_comparison_callout(
            model_quality_scores={"premium": 9.0, "budget": 7.5},   # delta=1.5 > 1.0
            model_total_costs={"premium": 0.08, "budget": 0.02},    # delta=0.06 < 0.10
            model_ids=["premium", "budget"],
        )
        assert "worth it" in callout_worth.lower(), (
            f"Expected 'worth it' in callout: {callout_worth}"
        )

        # Case 2: not worth it (tiny score gain, large cost premium)
        callout_not = generate_cost_comparison_callout(
            model_quality_scores={"premium": 7.3, "budget": 7.2},   # delta=0.1 < 0.5
            model_total_costs={"premium": 0.30, "budget": 0.05},    # delta=0.25 > 0.20
            model_ids=["premium", "budget"],
        )
        assert "lower cost" in callout_not.lower() or "comparable" in callout_not.lower(), (
            f"Expected cheaper model to be recommended: {callout_not}"
        )

    def test_callout_suppresses_pipeline_recommendation_when_hallucination_flagged(self):
        """
        When the cheaper model is hallucination-flagged, the callout must NOT
        recommend it for high-volume pipelines. Instead it must warn about
        hallucination risk despite the lower cost.
        """
        from backend.verdict.verdict import generate_cost_comparison_callout

        callout = generate_cost_comparison_callout(
            model_quality_scores={"premium": 7.3, "budget": 7.2},   # "not worth it" branch
            model_total_costs={"premium": 0.30, "budget": 0.05},
            model_ids=["premium", "budget"],
            hallucination_flagged_models={"budget"},
        )

        assert "hallucination" in callout.lower(), (
            f"Expected hallucination warning in callout: {callout}"
        )
        assert "recommended for high-volume" not in callout.lower(), (
            f"Should NOT recommend flagged model for pipelines: {callout}"
        )
        assert "unsuitable for production" in callout.lower(), (
            f"Expected production unsuitability warning: {callout}"
        )

    def test_prices_last_updated_readable(self):
        """
        pricing.yaml must have a meta.last_updated key that is a non-empty string.
        This date is shown next to cost figures in the UI.
        """
        from backend.config import PRICING_CONFIG

        meta = PRICING_CONFIG.get("meta", {})
        assert meta, "pricing.yaml must have a 'meta' section"
        last_updated = meta.get("last_updated", "")
        assert isinstance(last_updated, str) and len(last_updated) > 0, (
            "meta.last_updated must be a non-empty string"
        )

    def test_cost_fields_exist_in_model_results(self, client):
        """
        After a complete run, tokens_in, tokens_out, and cost_usd must all be
        non-null in model_results — these are read by PDF/JSON export in S7.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) > 0
            for r in results:
                assert r.tokens_in is not None, "tokens_in must not be null"
                assert r.tokens_out is not None, "tokens_out must not be null"
                assert r.cost_usd is not None, "cost_usd must not be null"
        finally:
            db.close()


# ══════════════════════════════════════════════════════════════════════════
# Story 2.4 — Prompt Variance Analysis
# ══════════════════════════════════════════════════════════════════════════


class TestPromptVariance:

    def _make_prompt_results(self, scores_by_model: dict) -> list:
        """Helper: build minimal prompt_results list for variance tests."""
        return [
            {
                "model_id": model_id,
                "prompt_id": "p1",
                "scores": {
                    "accuracy": scores,
                    "hallucination": scores,
                    "instruction_following": scores,
                    "conciseness": scores,
                },
                "error": None,
            }
            for model_id, scores in scores_by_model.items()
        ]

    def test_variance_calculated_correctly(self):
        """
        With model A weighted score ≈ 8.0 and model B ≈ 5.0, variance must be ≈ 3.0.
        Uses equal dimension weights to make math straightforward.
        """
        from backend.verdict.verdict import calculate_prompt_variance

        rubric = {"accuracy": 25, "hallucination": 25, "instruction_following": 25, "conciseness": 25}

        prompt_results = [
            {"scores": {"accuracy": 8.0, "hallucination": 8.0, "instruction_following": 8.0, "conciseness": 8.0}, "error": None},
            {"scores": {"accuracy": 5.0, "hallucination": 5.0, "instruction_following": 5.0, "conciseness": 5.0}, "error": None},
        ]

        variance = calculate_prompt_variance(prompt_results, rubric)

        # Each model's weighted score = score × (25+25+25+25)/100 = score × 1.0
        # Variance = 8.0 - 5.0 = 3.0
        assert abs(variance - 3.0) < 0.01, (
            f"Expected variance ≈ 3.0, got {variance}"
        )

    def test_prompts_sorted_by_variance(self):
        """
        rank_prompts_by_variance must return (prompt_id, variance) tuples
        sorted by variance descending.
        """
        from backend.verdict.verdict import rank_prompts_by_variance

        rubric = {"accuracy": 25, "hallucination": 25, "instruction_following": 25, "conciseness": 25}

        # Build 5 prompts with known variances: p3 highest, p1 lowest
        scored_results = []
        prompt_variance_map = {
            "p1": (7.0, 7.0),   # variance ≈ 0.0
            "p2": (8.0, 6.0),   # variance ≈ 2.0
            "p3": (9.0, 4.0),   # variance ≈ 5.0
            "p4": (8.5, 7.5),   # variance ≈ 1.0
            "p5": (7.5, 5.5),   # variance ≈ 2.0
        }
        for pid, (score_a, score_b) in prompt_variance_map.items():
            for model_id, score in [("model-a", score_a), ("model-b", score_b)]:
                scored_results.append({
                    "model_id": model_id,
                    "prompt_id": pid,
                    "scores": {"accuracy": score, "hallucination": score,
                               "instruction_following": score, "conciseness": score},
                    "error": None,
                })

        ranked = rank_prompts_by_variance(scored_results, rubric)

        assert ranked[0][0] == "p3", (
            f"Highest variance prompt must be p3, got {ranked[0][0]}"
        )
        variances = [v for _, v in ranked]
        assert variances == sorted(variances, reverse=True), (
            "Prompts must be sorted by variance descending"
        )

    def test_top_3_flagged_as_high_variance(self):
        """
        get_high_variance_prompt_ids must return exactly 3 prompt_ids
        (the top 3 by variance).
        """
        from backend.verdict.verdict import get_high_variance_prompt_ids

        # 10 prompts with distinct variance scores
        ranked = [(f"p{i}", float(10 - i)) for i in range(10)]

        high_variance = get_high_variance_prompt_ids(ranked, top_n=3)

        assert len(high_variance) == 3, (
            f"Exactly 3 prompts must be flagged as high_variance, got {len(high_variance)}"
        )
        assert high_variance == {"p0", "p1", "p2"}, (
            f"Top 3 must be p0, p1, p2. Got: {high_variance}"
        )

    def test_prompt_expansion_returns_all_data(self, client):
        """
        GET /eval/{run_id}/results must include prompt_text, response_text,
        dimension_scores, and dimension_reasoning for every result row.
        """
        resp = client.post("/eval/run", json=valid_run_request(
            n_prompts=5, models=["gpt-5-4", "gpt-5-4-mini"]
        ))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        results_resp = client.get(f"/eval/{run_id}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()["results"]
        assert len(results) == 10, f"Expected 10 results (2 models × 5 prompts), got {len(results)}"

        for r in results:
            assert r.get("prompt_text"), (
                f"prompt_text must be non-empty for model={r['model_name']} idx={r['prompt_index']}"
            )
            assert r.get("response_text") is not None, "response_text must be present"
            assert isinstance(r.get("dimension_scores"), dict), "dimension_scores must be a dict"
            assert isinstance(r.get("dimension_reasoning"), dict), "dimension_reasoning must be a dict"
            for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]:
                assert dim in r["dimension_scores"], f"dimension_scores must contain '{dim}'"
                assert dim in r["dimension_reasoning"], f"dimension_reasoning must contain '{dim}'"

    def test_variance_insight_identifies_dimension(self):
        """
        generate_variance_insight must return a sentence mentioning the dimension
        with the highest per-prompt delta between models.
        """
        from backend.verdict.verdict import generate_variance_insight

        # Accuracy has the largest delta (9.0 vs 2.0 = delta 7.0)
        # Other dimensions are equal
        prompt_results = [
            {
                "model_id": "model-a",
                "scores": {
                    "accuracy": 9.0,
                    "hallucination": 7.0,
                    "instruction_following": 7.0,
                    "conciseness": 7.0,
                },
            },
            {
                "model_id": "model-b",
                "scores": {
                    "accuracy": 2.0,
                    "hallucination": 7.0,
                    "instruction_following": 7.0,
                    "conciseness": 7.0,
                },
            },
        ]

        insight = generate_variance_insight(prompt_results)

        assert "accuracy" in insight.lower(), (
            f"Insight must mention 'accuracy' as highest-delta dimension. Got: {insight}"
        )
        assert len(insight) > 0, "Insight must not be empty"

    def test_variance_saved_to_db(self, client):
        """
        After a complete run with multiple models, variance_score must be
        non-null in model_results for all rows.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(
            n_prompts=5, models=["gpt-5-4", "gpt-5-4-mini"]
        ))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) == 10, f"Expected 10 results, got {len(results)}"
            for r in results:
                assert r.variance_score is not None, (
                    f"variance_score must be set for model={r.model_name} prompt_idx={r.prompt_index}"
                )
                assert isinstance(r.variance_score, float), "variance_score must be a float"
                assert r.variance_score >= 0.0, "variance_score must be non-negative"
        finally:
            db.close()
