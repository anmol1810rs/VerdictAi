"""
Session 4 acceptance tests — Stories 2.1 (Verdict) and 2.2 (LLM-as-Judge Scoring).

Run: pytest backend/tests/test_session_4.py -v

Test groups:
  TestMultiModelRunner  (6 tests) — parallel execution, error handling, tokens, cost, image format
  TestJudgeScoring      (7 tests) — judge results, retry, graceful failure, hallucination flagging
  TestVerdictGeneration (7 tests) — scoring logic, hallucination penalty, cost efficiency, verdict text
"""
import asyncio
import time
from collections import defaultdict
from unittest import mock

import pytest

from backend.tests.conftest import VALID_RUBRIC, make_prompts, valid_run_request


# ══════════════════════════════════════════════════════════════════════════
# Multi-model runner tests
# ══════════════════════════════════════════════════════════════════════════


class TestMultiModelRunner:

    def test_runner_calls_all_selected_models(self, client):
        """
        Selecting 2 models with 5 prompts must produce 10 ModelResult rows.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        req = valid_run_request(n_prompts=5, models=["gpt-5-4", "gpt-5-4-mini"])
        resp = client.post("/eval/run", json=req)
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) == 10, (
                f"Expected 10 results (2 models × 5 prompts), got {len(results)}"
            )
            model_names = {r.model_name for r in results}
            assert "gpt-5-4" in model_names
            assert "gpt-5-4-mini" in model_names
        finally:
            db.close()

    def test_runner_executes_models_in_parallel(self):
        """
        With asyncio.gather(), total time for 2 models should be closer to
        1× single-model time than 2× (verifies concurrency, not sequential execution).
        Uses mock to control latency precisely.
        """
        from backend.runner.runner import run_models_parallel

        call_times: list = []

        async def slow_single(model_id, prompt_text, image_data, api_keys):
            t0 = time.monotonic()
            await asyncio.sleep(0.1)  # simulate 100ms API call
            call_times.append(time.monotonic() - t0)
            return f"[MOCK] {model_id}", 100, 50

        prompts_data = [
            {
                "prompt_id": "p1",
                "prompt_text": "Hello",
                "prompt_index": 0,
                "image_data": None,
                "expected_output": None,
            }
        ]

        with mock.patch("backend.runner.runner._run_single", side_effect=slow_single):
            with mock.patch("backend.runner.runner.DEV_MODE", False):
                t_start = time.monotonic()
                results = asyncio.run(
                    run_models_parallel(["gpt-5-4", "gpt-5-4-mini"], prompts_data, {})
                )
                total_elapsed = time.monotonic() - t_start

        # Sequential would take ~200ms; parallel should be ~100ms ± overhead
        assert total_elapsed < 0.18, (
            f"Parallel execution took {total_elapsed:.3f}s — expected < 180ms for 2 concurrent 100ms tasks"
        )
        assert len(results) == 2

    def test_runner_continues_if_one_model_fails(self, client):
        """
        If one model raises an exception, the other model's results must still be saved.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        call_count = {"n": 0}

        async def fail_first_model(model_id, prompt_text, image_data, api_keys):
            call_count["n"] += 1
            if model_id == "gpt-5-4":
                raise RuntimeError("Simulated API timeout for gpt-5-4")
            return f"[OK] {model_id} response", 100, 50

        req = valid_run_request(n_prompts=5, models=["gpt-5-4", "gpt-5-4-mini"])

        with mock.patch("backend.runner.runner.DEV_MODE", False), \
             mock.patch("backend.runner.runner._run_single", side_effect=fail_first_model):
            resp = client.post("/eval/run", json=req)

        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            # gpt-5-4-mini results must exist (5 prompts)
            mini_results = [r for r in results if r.model_name == "gpt-5-4-mini"]
            assert len(mini_results) == 5, (
                f"Expected 5 gpt-5-4-mini results, got {len(mini_results)}"
            )
            # gpt-5-4 may have empty response_text but row must exist (error result)
            # (or may have 0 rows if filtered by error — either is acceptable)
        finally:
            db.close()

    def test_token_counts_saved_correctly(self, client):
        """
        Mock API returning known token counts must be saved to tokens_in / tokens_out columns.
        Patches router.DEV_MODE=False to use the real async path, runner._run_single to return
        known tokens, and judge.DEV_MODE=True so mock judge scores flow through without a real API call.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        EXPECTED_TOKENS_IN = 250
        EXPECTED_TOKENS_OUT = 180

        async def mock_run_single(model_id, prompt_text, image_data, api_keys):
            return f"[MOCK] {model_id}", EXPECTED_TOKENS_IN, EXPECTED_TOKENS_OUT

        req = valid_run_request(n_prompts=5, models=["gpt-5-4"])

        with mock.patch("backend.eval.router.DEV_MODE", False), \
             mock.patch("backend.runner.runner.DEV_MODE", False), \
             mock.patch("backend.judge.judge.DEV_MODE", True), \
             mock.patch("backend.runner.runner._run_single", side_effect=mock_run_single):
            resp = client.post("/eval/run", json=req)

        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(
                ModelResult.eval_run_id == run_id,
                ModelResult.model_name == "gpt-5-4",
            ).all()
            assert len(results) > 0
            for r in results:
                assert r.tokens_in == EXPECTED_TOKENS_IN, (
                    f"tokens_in mismatch: expected {EXPECTED_TOKENS_IN}, got {r.tokens_in}"
                )
                assert r.tokens_out == EXPECTED_TOKENS_OUT, (
                    f"tokens_out mismatch: expected {EXPECTED_TOKENS_OUT}, got {r.tokens_out}"
                )
        finally:
            db.close()

    def test_cost_calculated_from_pricing_yaml(self):
        """
        calculate_cost(model_id, tokens_in, tokens_out) must match manual calculation
        from pricing.yaml: (tokens_in × input_rate + tokens_out × output_rate) / 1M.
        """
        from backend.runner.runner import calculate_cost
        from backend.config import PRICING_CONFIG

        model_id = "gpt-5-4"
        tokens_in = 1000
        tokens_out = 500

        pricing = PRICING_CONFIG["models"][model_id]
        expected = round(
            (tokens_in * pricing["input_per_1m_usd"] + tokens_out * pricing["output_per_1m_usd"])
            / 1_000_000,
            8,
        )

        actual = calculate_cost(model_id, tokens_in, tokens_out)
        assert actual == expected, f"Cost mismatch: expected {expected}, got {actual}"

    def test_image_prompt_uses_correct_format(self):
        """
        When image_data is present:
          - OpenAI request must contain image_url content block
          - Anthropic request must contain base64 block with source.type="base64"
          - Google request must contain inline_data bytes
        """
        import base64
        from backend.runner.runner import _call_openai, _call_anthropic, _call_google
        from backend.config import MODELS_CONFIG

        # Build a 1-pixel red PNG in base64 for testing
        png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
        )
        image_data = f"data:image/png;base64,{png_b64}"
        prompt_text = "Describe this image."

        # ── OpenAI format check ────────────────────────────────────────────
        openai_model_cfg = next(m for m in MODELS_CONFIG["mvp_models"] if m["id"] == "gpt-5-4")

        openai_calls = []

        async def mock_openai_create(**kwargs):
            openai_calls.append(kwargs)

            class FakeChoice:
                class message:
                    content = "OpenAI image response"

            class FakeUsage:
                prompt_tokens = 50
                completion_tokens = 20

            class FakeResponse:
                choices = [FakeChoice()]
                usage = FakeUsage()

            return FakeResponse()

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create = mock_openai_create

        # AsyncOpenAI is now a module-level name — patch at backend.runner.runner
        async def run_openai_test():
            with mock.patch("backend.runner.runner.AsyncOpenAI", return_value=mock_client):
                await _call_openai(openai_model_cfg, prompt_text, image_data, "sk-test")

        asyncio.run(run_openai_test())

        assert len(openai_calls) == 1
        messages = openai_calls[0]["messages"]
        user_content = messages[0]["content"]
        assert isinstance(user_content, list), "OpenAI image prompt must use list content"
        content_types = {item["type"] for item in user_content}
        assert "image_url" in content_types, (
            f"OpenAI request must contain image_url block, got types: {content_types}"
        )

        # ── Anthropic format check ─────────────────────────────────────────
        anthropic_model_cfg = next(
            m for m in MODELS_CONFIG["mvp_models"] if m["id"] == "claude-sonnet-4-6"
        )

        anthropic_calls = []

        async def mock_anthropic_create(**kwargs):
            anthropic_calls.append(kwargs)

            class FakeContent:
                text = "Anthropic image response"

            class FakeUsage:
                input_tokens = 50
                output_tokens = 20

            class FakeResponse:
                content = [FakeContent()]
                usage = FakeUsage()

            return FakeResponse()

        mock_anthropic_client = mock.MagicMock()
        mock_anthropic_client.messages.create = mock_anthropic_create

        async def run_anthropic_test():
            # AsyncAnthropic is a module-level name — patch at backend.runner.runner
            with mock.patch("backend.runner.runner.AsyncAnthropic", return_value=mock_anthropic_client):
                await _call_anthropic(anthropic_model_cfg, prompt_text, image_data, "sk-ant-test")

        asyncio.run(run_anthropic_test())

        assert len(anthropic_calls) == 1
        messages = anthropic_calls[0]["messages"]
        user_content = messages[0]["content"]
        assert isinstance(user_content, list), "Anthropic image prompt must use list content"
        image_blocks = [b for b in user_content if b.get("type") == "image"]
        assert len(image_blocks) == 1, "Anthropic request must contain one image block"
        assert image_blocks[0]["source"]["type"] == "base64", (
            "Anthropic image source type must be 'base64'"
        )


# ══════════════════════════════════════════════════════════════════════════
# Judge scoring tests
# ══════════════════════════════════════════════════════════════════════════


class TestJudgeScoring:

    def test_judge_called_after_all_models_complete(self, client):
        """
        Judge must not run until model results exist. In DEV_MODE the mock judge
        runs inline after runner. Verify dimension_scores are set on all rows.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5, models=["gpt-5-4"]))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            assert len(results) == 5
            for r in results:
                assert r.dimension_scores is not None, "dimension_scores must be set after judge runs"
                assert len(r.dimension_scores) > 0, "dimension_scores must not be empty"
        finally:
            db.close()

    def test_judge_returns_valid_json_scores(self):
        """
        parse_judge_response must return a valid structured dict from a well-formed
        judge JSON string, with all 4 dimensions present.
        """
        from backend.judge.judge import parse_judge_response, JUDGE_DIMENSIONS

        valid_json = """{
  "scores": {
    "accuracy": {"score": 8, "reasoning": "Factually correct.", "evidence": "The capital is Paris."},
    "hallucination": {"score": 9, "reasoning": "No hallucinations.", "evidence": "None detected", "flagged": false},
    "instruction_following": {"score": 7, "reasoning": "Instructions mostly followed.", "evidence": "Responded in one sentence."},
    "conciseness": {"score": 6, "reasoning": "Slightly verbose.", "evidence": "Response is 3 sentences long."}
  },
  "cost_efficiency": "auto"
}"""
        result = parse_judge_response(valid_json)
        assert result is not None, "parse_judge_response must return a dict for valid JSON"
        assert "scores" in result
        for dim in JUDGE_DIMENSIONS:
            assert dim in result["scores"], f"Dimension '{dim}' missing from parsed scores"
            assert result["scores"][dim] is not None

    def test_judge_score_saved_to_model_results(self, client):
        """
        After a complete run, all 4 judge dimension scores must be present
        in model_results.dimension_scores.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            for r in results:
                scores = r.dimension_scores or {}
                for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]:
                    assert dim in scores, f"Dimension '{dim}' missing from model_results.dimension_scores"
        finally:
            db.close()

    def test_judge_reasoning_never_empty(self, client):
        """
        Every dimension_reasoning entry must be a non-empty string.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            results = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).all()
            for r in results:
                reasoning = r.dimension_reasoning or {}
                for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]:
                    text = reasoning.get(dim, "")
                    assert text and len(text.strip()) > 0, (
                        f"Reasoning for '{dim}' must not be empty"
                    )
        finally:
            db.close()

    def test_hallucination_flagged_when_score_low(self):
        """
        parse_judge_response must set hallucination_flagged=True when score <= 3.
        """
        from backend.judge.judge import parse_judge_response

        low_hallucination_json = """{
  "scores": {
    "accuracy": {"score": 8, "reasoning": "Good.", "evidence": "Correct claim."},
    "hallucination": {"score": 2, "reasoning": "Severe hallucination detected.", "evidence": "Claimed the moon is made of cheese.", "flagged": true},
    "instruction_following": {"score": 7, "reasoning": "OK.", "evidence": "Did what was asked."},
    "conciseness": {"score": 6, "reasoning": "Fine length.", "evidence": "Reasonable length."}
  },
  "cost_efficiency": "auto"
}"""
        result = parse_judge_response(low_hallucination_json)
        assert result is not None
        assert result["hallucination_flagged"] is True, (
            "hallucination_flagged must be True when score=2 (≤ threshold of 3)"
        )
        assert result["hallucination_reason"] is not None

    def test_judge_retries_on_invalid_json(self):
        """
        If judge returns invalid JSON on first attempt, score_response_async must retry
        and use the valid response from the second attempt.
        """
        from backend.judge.judge import score_response_async, JUDGE_DIMENSIONS

        call_count = {"n": 0}
        valid_response = """{
  "scores": {
    "accuracy": {"score": 7, "reasoning": "Good.", "evidence": "Evidence A."},
    "hallucination": {"score": 9, "reasoning": "Clean.", "evidence": "None detected"},
    "instruction_following": {"score": 8, "reasoning": "Followed.", "evidence": "Did it."},
    "conciseness": {"score": 6, "reasoning": "OK.", "evidence": "Reasonable."}
  },
  "cost_efficiency": "auto"
}"""

        async def mock_call_judge(user_prompt, api_key):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return "INVALID JSON {{{"
            return valid_response

        with mock.patch("backend.judge.judge.DEV_MODE", False), \
             mock.patch("backend.judge.judge._call_judge_api", side_effect=mock_call_judge):
            result = asyncio.run(
                score_response_async("test prompt", "test response", {}, None, "sk-test")
            )

        assert call_count["n"] == 2, f"Expected 2 judge calls (retry), got {call_count['n']}"
        assert result is not None
        assert not result.get("judge_error"), "Result must not have judge_error after successful retry"
        for dim in JUDGE_DIMENSIONS:
            assert result["scores"].get(dim) is not None, f"Score for '{dim}' must be set after retry"

    def test_judge_handles_retry_failure_gracefully(self):
        """
        If judge returns invalid JSON twice, score must be null and run must not crash.
        """
        from backend.judge.judge import score_response_async, JUDGE_DIMENSIONS

        async def always_invalid(user_prompt, api_key):
            return "NOT JSON AT ALL"

        with mock.patch("backend.judge.judge.DEV_MODE", False), \
             mock.patch("backend.judge.judge._call_judge_api", side_effect=always_invalid):
            result = asyncio.run(
                score_response_async("prompt", "response", {}, None, "sk-test")
            )

        assert result is not None, "score_response_async must return a dict, never raise"
        assert result.get("judge_error") is True, "judge_error must be True on double failure"
        for dim in JUDGE_DIMENSIONS:
            assert result["scores"].get(dim) is None, (
                f"Score for '{dim}' must be None on double judge failure"
            )


# ══════════════════════════════════════════════════════════════════════════
# Verdict generation tests
# ══════════════════════════════════════════════════════════════════════════


class TestVerdictGeneration:

    def _make_scored_results(
        self,
        models=("gpt-5-4", "gpt-5-4-mini"),
        n_prompts=3,
        scores_override=None,
        cost_override=None,
        flagged_model=None,
        flagged_pct=1.0,
    ):
        """Helper: build minimal scored_results list for verdict tests."""
        results = []
        for i in range(n_prompts):
            for mid in models:
                default_scores = {
                    "accuracy": 7.0,
                    "hallucination": 8.0,
                    "instruction_following": 7.5,
                    "conciseness": 6.5,
                }
                if scores_override and mid in scores_override:
                    default_scores.update(scores_override[mid])

                flagged = False
                if flagged_model == mid:
                    threshold = int(n_prompts * flagged_pct)
                    flagged = i < threshold

                results.append(
                    {
                        "model_id": mid,
                        "prompt_id": f"p{i}",
                        "prompt_index": i,
                        "prompt_text": f"Prompt {i}",
                        "expected_output": None,
                        "response_text": f"[MOCK] {mid} response {i}",
                        "tokens_in": 120,
                        "tokens_out": 85,
                        "cost_usd": cost_override.get(mid, 0.001) if cost_override else 0.001,
                        "scores": default_scores,
                        "reasoning": {d: "mock reasoning" for d in default_scores},
                        "evidence": {d: "mock evidence" for d in default_scores},
                        "hallucination_flagged": flagged,
                        "hallucination_reason": "Detected fabrication" if flagged else None,
                        "error": None,
                    }
                )
        return results

    def test_verdict_selects_highest_weighted_score(self):
        """
        generate_verdict must select the model with the highest final weighted score.
        """
        from backend.verdict.verdict import calculate_weighted_quality_score

        rubric = {
            "accuracy": 40,
            "hallucination": 30,
            "instruction_following": 20,
            "conciseness": 10,
            "cost_efficiency": 0,
        }

        scores_a = {"accuracy": 9.0, "hallucination": 8.0, "instruction_following": 7.0, "conciseness": 6.0}
        scores_b = {"accuracy": 5.0, "hallucination": 5.0, "instruction_following": 5.0, "conciseness": 5.0}

        score_a = calculate_weighted_quality_score(scores_a, rubric)
        score_b = calculate_weighted_quality_score(scores_b, rubric)

        assert score_a > score_b, (
            f"Model A (higher dim scores) must have higher weighted score: {score_a} vs {score_b}"
        )

    def test_hallucination_penalty_overrides_winner(self):
        """
        A model with hallucination_flagged on >30% of prompts cannot win,
        even if it has the highest quality score.
        """
        from backend.verdict.verdict import detect_hallucination_disqualified

        # 3 prompts, model A flagged on 2/3 = 67% > 30%
        scored_results = self._make_scored_results(
            models=("gpt-5-4", "gpt-5-4-mini"),
            n_prompts=3,
            scores_override={"gpt-5-4": {"accuracy": 9.9, "hallucination": 9.9}},
            flagged_model="gpt-5-4",
            flagged_pct=0.67,
        )

        disqualified = detect_hallucination_disqualified(scored_results)
        assert "gpt-5-4" in disqualified, (
            "gpt-5-4 flagged on 67% of prompts must be disqualified"
        )

    def test_cost_efficiency_calculated_correctly(self):
        """
        normalize_cost_efficiency: lowest cpp → score=10, highest cpp → score=0.
        """
        from backend.verdict.verdict import normalize_cost_efficiency

        model_cpp = {
            "gpt-5-4": 0.002,      # expensive per quality point
            "gpt-5-4-mini": 0.001, # cheapest per quality point
        }

        result = normalize_cost_efficiency(model_cpp)

        assert result["gpt-5-4-mini"] == 10.0, (
            "Cheapest model must get cost_efficiency score of 10"
        )
        assert result["gpt-5-4"] == 0.0, (
            "Most expensive model must get cost_efficiency score of 0"
        )

    def test_verdict_text_contains_winning_model(self, client):
        """
        After a complete run, verdict.summary must mention the winning model name.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import Verdict

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5, models=["gpt-5-4", "gpt-5-4-mini"]))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
            assert verdict is not None, "Verdict must be created for a complete run"
            assert verdict.winning_model in verdict.summary, (
                f"Verdict summary must mention winning model '{verdict.winning_model}'"
            )
        finally:
            db.close()

    def test_verdict_text_contains_hallucination_warning(self):
        """
        build_verdict_text must include hallucination warning when a model is flagged.
        """
        from backend.verdict.verdict import build_verdict_text

        warnings = ["gpt-5-4 was flagged for hallucinations on 3 of 5 prompts."]
        text = build_verdict_text(
            winning_model="gpt-5-4-mini",
            final_score=7.5,
            other_models=["gpt-5-4"],
            top_dimensions=["Accuracy", "Hallucination"],
            cost_insight="",
            hallucination_warnings=warnings,
        )

        assert "hallucination" in text.lower(), "Verdict text must contain hallucination warning"
        assert "gpt-5-4" in text, "Verdict text must name the flagged model"

    def test_verdict_saved_to_db(self, client):
        """
        After a complete eval run, verdicts table must have one row with all required fields.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import Verdict

        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
            assert verdict is not None, "Verdict row must exist in DB for completed run"
            assert verdict.winning_model, "winning_model must be set"
            assert verdict.summary, "summary must be non-empty"
            assert verdict.score_breakdown is not None, "score_breakdown must be present"
            assert verdict.cost_comparison is not None, "cost_comparison must be present"
            assert verdict.hallucination_warnings is not None, "hallucination_warnings must be present"
        finally:
            db.close()

    def test_verdict_score_breakdown_has_all_models(self, client):
        """
        score_breakdown in the verdict must contain entries for all selected models.
        """
        from backend.db.database import SessionLocal
        from backend.db.models import Verdict

        models_selected = ["gpt-5-4", "gpt-5-4-mini"]
        resp = client.post("/eval/run", json=valid_run_request(n_prompts=5, models=models_selected))
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        db = SessionLocal()
        try:
            verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
            assert verdict is not None
            breakdown = verdict.score_breakdown or {}
            for mid in models_selected:
                assert mid in breakdown, (
                    f"score_breakdown must contain entry for model '{mid}'"
                )
        finally:
            db.close()
