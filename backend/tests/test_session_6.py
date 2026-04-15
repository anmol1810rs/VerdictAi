"""
Session 6 acceptance tests — Stories 2.5 (Ground Truth Comparison) and
2.6 (Historical Run Comparison).

Run: pytest backend/tests/test_session_6.py -v

Test groups:
  TestGroundTruth   (6 tests) — GT score/reasoning saved, null when absent,
                                verdict mentions alignment, score range valid
  TestRunCompare    (7 tests) — compare endpoint, deltas, winner change,
                                different models, insight, error on incomplete run,
                                cost delta
"""
import pytest

from backend.tests.conftest import VALID_RUBRIC, make_prompts, valid_run_request


# ══════════════════════════════════════════════════════════════════════════
# Story 2.5 — Ground Truth Comparison
# ══════════════════════════════════════════════════════════════════════════


class TestGroundTruth:

    def test_gt_score_calculated_when_gt_provided(self, client):
        """
        When expected_output is present, ground_truth_score must be non-null
        in the DB after a completed run.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        req = {
            "prompts": make_prompts(5, with_gt=True),
            "models_selected": ["gpt-5-4"],
            "rubric": VALID_RUBRIC,
            "api_keys": {"openai_api_key": "sk-test1234567890abcdefghij"},
        }
        resp = client.post("/eval/run", json=req)
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) > 0
            for r in results:
                assert r.ground_truth_score is not None, (
                    f"ground_truth_score must not be null when GT provided (result id={r.id})"
                )
        finally:
            db.close()

    def test_gt_score_null_when_no_gt(self, client):
        """
        When no expected_output is provided, ground_truth_score must be null
        and the run must complete without errors.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import EvalRun, ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
            assert run.status == "complete", f"Run failed: {run.error_message}"

            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            for r in results:
                assert r.ground_truth_score is None, (
                    "ground_truth_score must be null when no GT is provided"
                )
        finally:
            db.close()

    def test_gt_reasoning_saved_to_db(self, client):
        """
        When expected_output is present, ground_truth_reasoning must be
        a non-empty string saved to model_results.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        req = {
            "prompts": make_prompts(5, with_gt=True),
            "models_selected": ["gpt-5-4"],
            "rubric": VALID_RUBRIC,
            "api_keys": {"openai_api_key": "sk-test1234567890abcdefghij"},
        }
        resp = client.post("/eval/run", json=req)
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            for r in results:
                assert r.ground_truth_reasoning is not None, (
                    "ground_truth_reasoning must not be null when GT provided"
                )
                assert len(r.ground_truth_reasoning) > 0, (
                    "ground_truth_reasoning must be a non-empty string"
                )
        finally:
            db.close()

    def test_gt_column_absent_when_no_gt(self, client):
        """
        When no GT provided, the results API response must carry
        ground_truth_score=null (null = not shown in UI).
        """
        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        run_id = resp.json()["run_id"]

        results_resp = client.get(f"/eval/{run_id}/results")
        assert results_resp.status_code == 200

        for result in results_resp.json()["results"]:
            assert result.get("ground_truth_score") is None, (
                "ground_truth_score must be null (not shown) when no GT provided"
            )

    def test_gt_alignment_in_verdict_when_present(self, client):
        """
        When GT is provided, the verdict summary must mention 'alignment'
        (the GT alignment sentence added by build_gt_alignment_summary).
        """
        req = {
            "prompts": make_prompts(5, with_gt=True),
            "models_selected": ["gpt-5-4"],
            "rubric": VALID_RUBRIC,
            "api_keys": {"openai_api_key": "sk-test1234567890abcdefghij"},
        }
        resp = client.post("/eval/run", json=req)
        run_id = resp.json()["run_id"]

        results_resp = client.get(f"/eval/{run_id}/results")
        assert results_resp.status_code == 200

        verdict = results_resp.json().get("verdict", {})
        summary = verdict.get("summary", "")
        assert "alignment" in summary.lower(), (
            f"Verdict summary must mention 'alignment' when GT provided. Got: {summary}"
        )

    def test_gt_score_range_valid(self, client):
        """Ground truth scores must be in the 0-10 range."""
        from backend.judge.mock_judge import get_mock_gt_score

        # Test the mock helper directly for the range check
        score, _ = get_mock_gt_score("Paris")
        assert score is not None
        assert 0.0 <= score <= 10.0, f"GT score {score} is out of 0-10 range"

        # Also confirm null when no GT
        score_none, reason_none = get_mock_gt_score(None)
        assert score_none is None
        assert reason_none is None


# ══════════════════════════════════════════════════════════════════════════
# Story 2.6 — Historical Run Comparison
# ══════════════════════════════════════════════════════════════════════════


def _make_completed_run(client, models=None, n_prompts=5, label=None):
    """Helper: create and return the run_id of a completed eval run."""
    req = valid_run_request(n_prompts=n_prompts, models=models or ["gpt-5-4"])
    if label:
        req["custom_label"] = label
    resp = client.post("/eval/run", json=req)
    assert resp.status_code == 200
    return resp.json()["run_id"]


class TestRunCompare:

    def test_compare_endpoint_returns_both_runs(self, client):
        """
        /eval/compare must return run_a and run_b data with their IDs.
        """
        run_a = _make_completed_run(client, label="Compare-A")
        run_b = _make_completed_run(client, label="Compare-B")

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        data = resp.json()
        assert data["run_a"]["id"] == run_a
        assert data["run_b"]["id"] == run_b
        assert "deltas" in data

    def test_compare_calculates_score_delta(self, client):
        """
        score_delta must be calculated: run_b_score - run_a_score.
        Since both runs use DEV_MODE mock scores (identical), deltas should be ~0.
        """
        run_a = _make_completed_run(client)
        run_b = _make_completed_run(client)

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        deltas = resp.json()["deltas"]
        score_delta = deltas["score_delta"]

        # score_delta must be a dict with model keys
        assert isinstance(score_delta, dict), "score_delta must be a dict"
        for model, dim_deltas in score_delta.items():
            assert isinstance(dim_deltas, dict), f"dim_deltas for {model} must be dict"
            for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]:
                assert dim in dim_deltas, f"Dimension {dim} must be in score_delta"

    def test_compare_detects_winner_change(self, client):
        """
        winner_changed must reflect whether the winning model changed between runs.
        For identical mock runs, winner_changed should be False.
        """
        run_a = _make_completed_run(client)
        run_b = _make_completed_run(client)

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        deltas = resp.json()["deltas"]
        # Both runs use same DEV_MODE scores → same winner
        assert deltas["winner_changed"] is False
        assert "winner_changed" in deltas

    def test_compare_handles_different_models(self, client):
        """
        When runs use different models, score_delta for non-overlapping models
        must have None deltas (not an error).
        """
        run_a = _make_completed_run(client, models=["gpt-5-4"])
        run_b = _make_completed_run(client, models=["gpt-5-4-mini"])

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        score_delta = resp.json()["deltas"]["score_delta"]
        cost_delta = resp.json()["deltas"]["cost_delta"]

        # gpt-5-4 is only in run_a → deltas should be None
        if "gpt-5-4" in score_delta:
            for dim_val in score_delta["gpt-5-4"].values():
                assert dim_val is None, "Non-overlapping model must have None deltas"
            assert cost_delta.get("gpt-5-4") is None

        # gpt-5-4-mini is only in run_b → deltas should be None
        if "gpt-5-4-mini" in score_delta:
            for dim_val in score_delta["gpt-5-4-mini"].values():
                assert dim_val is None, "Non-overlapping model must have None deltas"

    def test_compare_insight_generated(self, client):
        """
        The compare response must include a non-empty insight sentence in deltas.
        """
        run_a = _make_completed_run(client)
        run_b = _make_completed_run(client)

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        insight = resp.json()["deltas"].get("insight", "")
        assert isinstance(insight, str) and len(insight) > 0, (
            f"insight must be a non-empty string, got: {insight!r}"
        )

    def test_compare_requires_two_completed_runs(self, client):
        """
        Calling compare with a non-existent run ID must return 404.
        """
        run_a = _make_completed_run(client)
        fake_id = "00000000-0000-0000-0000-000000000000"

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": fake_id})
        assert resp.status_code == 404, (
            f"Expected 404 for unknown run, got {resp.status_code}: {resp.text}"
        )

    def test_compare_cost_delta_correct(self, client):
        """
        cost_delta per model must be present in the deltas dict.
        For two identical DEV_MODE runs cost_delta should be ~0.
        """
        run_a = _make_completed_run(client)
        run_b = _make_completed_run(client)

        resp = client.get("/eval/compare", params={"run_a": run_a, "run_b": run_b})
        assert resp.status_code == 200

        cost_delta = resp.json()["deltas"]["cost_delta"]
        assert isinstance(cost_delta, dict), "cost_delta must be a dict"
        for model, delta in cost_delta.items():
            assert delta is not None, f"cost_delta for {model} must not be None in shared run"
            # For same mock pricing, expect ~0
            assert abs(delta) < 0.01, (
                f"Identical runs should have ~0 cost delta, got {delta} for {model}"
            )
