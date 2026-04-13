"""
Layer 2 — Session 1 story acceptance tests.

Covers Stories 1.1, 1.2, 1.3 acceptance criteria from the PRD.

Run: pytest backend/tests/test_session_1.py -v
"""
import io
import csv
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from backend.config import MODELS_CONFIG, RUBRIC_PRESETS
from backend.db.database import engine, get_table_names
from backend.judge.mock_judge import MOCK_SCORES
from backend.tests.conftest import (
    VALID_API_KEYS,
    VALID_RUBRIC,
    make_prompts,
    valid_run_request,
)


# ── CSV helpers ────────────────────────────────────────────────────────────

def make_csv_bytes(rows: list[dict], columns: list[str] | None = None) -> bytes:
    if not rows:
        # Empty CSV — just header
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=columns or ["prompt"])
        writer.writeheader()
        return buf.getvalue().encode()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def make_jsonl_bytes(rows: list[dict]) -> bytes:
    return b"\n".join(json.dumps(r).encode() for r in rows)


def upload_csv(client: TestClient, rows: list[dict], columns: list[str] | None = None):
    content = make_csv_bytes(rows, columns)
    return client.post(
        "/upload",
        files={"file": ("dataset.csv", io.BytesIO(content), "text/csv")},
    )


def upload_jsonl(client: TestClient, rows: list[dict]):
    content = make_jsonl_bytes(rows)
    return client.post(
        "/upload",
        files={"file": ("dataset.jsonl", io.BytesIO(content), "application/json")},
    )


# ══════════════════════════════════════════════════════════════════════════
# STORY 1.1 — Dataset Upload
# PRD acceptance criteria: 1–6
# ══════════════════════════════════════════════════════════════════════════

class TestUploadCSV:
    def test_upload_accepts_valid_csv(self, client: TestClient):
        """AC1: Accepts CSV with prompt column. Returns 200 + correct prompt count."""
        rows = make_prompts(10)
        resp = upload_csv(client, rows)
        assert resp.status_code == 200
        body = resp.json()
        assert body["prompt_count"] == 10
        assert body["modality"] == "text"

    def test_upload_rejects_empty_csv(self, client: TestClient):
        """AC5: Minimum 5 prompts enforced. Empty file returns 422."""
        resp = upload_csv(client, [], columns=["prompt"])
        assert resp.status_code == 422
        assert "error" in resp.json() or "detail" in resp.json()

    def test_upload_enforces_100_prompt_max(self, client: TestClient):
        """AC5: Maximum 100 prompts enforced. 101 rows returns 422."""
        rows = make_prompts(101)
        resp = upload_csv(client, rows)
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "100" in detail or "maximum" in detail.lower()

    def test_upload_enforces_minimum_5_prompts(self, client: TestClient):
        """AC5: Fewer than 5 prompts returns 422."""
        rows = make_prompts(3)
        resp = upload_csv(client, rows)
        assert resp.status_code == 422

    def test_upload_accepts_100_prompts_exactly(self, client: TestClient):
        """AC5: Exactly 100 prompts is accepted."""
        rows = make_prompts(100)
        resp = upload_csv(client, rows)
        assert resp.status_code == 200
        assert resp.json()["prompt_count"] == 100

    def test_upload_accepts_optional_ground_truth(self, client: TestClient):
        """AC1: Optional expected_output column is parsed correctly."""
        rows = make_prompts(5, with_gt=True)
        resp = upload_csv(client, rows)
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_ground_truth"] is True
        # Verify actual content
        assert body["prompts"][0]["expected_output"] == "Expected answer 1"

    def test_upload_without_ground_truth_has_ground_truth_false(self, client: TestClient):
        """AC1: has_ground_truth is False when no expected_output column."""
        rows = make_prompts(5)
        resp = upload_csv(client, rows)
        assert resp.status_code == 200
        assert resp.json()["has_ground_truth"] is False

    def test_upload_accepts_optional_engineer_name(self, client: TestClient):
        """AC1: Optional engineer_name column is parsed correctly."""
        rows = make_prompts(5, with_engineer=True)
        resp = upload_csv(client, rows)
        assert resp.status_code == 200
        body = resp.json()
        assert body["has_engineer_names"] is True
        assert body["prompts"][0]["engineer_name"] == "Alice"

    def test_upload_rejects_missing_prompt_column(self, client: TestClient):
        """AC1: Rows without 'prompt' column return 422."""
        rows = [{"question": f"Q{i}"} for i in range(5)]
        resp = upload_csv(client, rows)
        assert resp.status_code == 422

    def test_upload_accepts_jsonl(self, client: TestClient):
        """AC1: JSONL format is accepted alongside CSV."""
        rows = make_prompts(5)
        resp = upload_jsonl(client, rows)
        assert resp.status_code == 200
        assert resp.json()["prompt_count"] == 5

    def test_upload_rejects_unsupported_format(self, client: TestClient):
        """Unsupported file type returns 422."""
        resp = client.post(
            "/upload",
            files={"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════
# STORY 1.2 — API Keys
# PRD acceptance criteria: 1–7
# ══════════════════════════════════════════════════════════════════════════

class TestAPIKeys:
    def test_api_keys_not_persisted_to_db(self, client: TestClient):
        """AC1: API keys must NEVER appear as columns in eval_runs table."""
        # Run an eval (keys are passed but must not be saved)
        resp = client.post("/eval/run", json=valid_run_request())
        assert resp.status_code == 200

        inspector = sa_inspect(engine)
        columns = [col["name"].lower() for col in inspector.get_columns("eval_runs")]
        forbidden = ["api_key", "openai_key", "anthropic_key", "google_key", "secret", "token"]
        for forbidden_col in forbidden:
            matching = [c for c in columns if forbidden_col in c]
            assert not matching, f"Found forbidden column '{matching}' in eval_runs table"

    def test_api_key_validation_catches_invalid_key(self, client: TestClient):
        """AC2, AC3: Invalid OpenAI key format returns 422 before eval starts."""
        resp = client.post("/keys/validate", json={"openai_api_key": "abc123"})
        assert resp.status_code == 422

    def test_api_key_validation_catches_wrong_prefix(self, client: TestClient):
        """Key without 'sk-' prefix is rejected."""
        resp = client.post("/keys/validate", json={"openai_api_key": "pk-thisisalongenoughkey123"})
        assert resp.status_code == 422

    def test_valid_openai_key_accepted(self, client: TestClient):
        """Valid-format OpenAI key passes validation."""
        resp = client.post("/keys/validate", json={"openai_api_key": "sk-test1234567890abcdefgh"})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True
        assert resp.json()["openai"] is True

    def test_optional_provider_keys_absent(self, client: TestClient):
        """AC2: Anthropic and Google keys are optional."""
        resp = client.post("/keys/validate", json={"openai_api_key": "sk-test1234567890abcdefgh"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["anthropic"] is False
        assert body["google"] is False

    def test_optional_provider_keys_present(self, client: TestClient):
        """When optional keys are provided they are reported as present."""
        resp = client.post("/keys/validate", json={
            "openai_api_key": "sk-test1234567890abcdefgh",
            "anthropic_api_key": "sk-ant-key12345",
            "google_api_key": "AIzaSy-googlekey12345",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["anthropic"] is True
        assert body["google"] is True

    def test_eval_run_never_stores_api_keys(self, client: TestClient):
        """AC1: eval_runs rows contain zero API key data."""
        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]

        # Fetch results and confirm no key data leaks
        results_resp = client.get(f"/eval/{run_id}/results")
        body = results_resp.json()
        body_str = json.dumps(body).lower()
        assert "sk-" not in body_str
        assert "api_key" not in body_str


# ══════════════════════════════════════════════════════════════════════════
# STORY 1.3 — Rubric Configurator
# PRD acceptance criteria: 1–5
# ══════════════════════════════════════════════════════════════════════════

class TestRubric:
    def test_rubric_preset_weights_sum_to_100(self):
        """AC1: All 3 presets must sum to exactly 100."""
        for preset_name, preset in RUBRIC_PRESETS.items():
            total = sum(preset["weights"].values())
            assert total == 100, (
                f"Preset '{preset_name}' weights sum to {total}, expected 100"
            )

    def test_three_presets_defined(self):
        """AC1: Exactly 3 presets: customer_support, technical_documentation, data_labeling_qa."""
        expected = {"customer_support", "technical_documentation", "data_labeling_qa"}
        assert set(RUBRIC_PRESETS.keys()) == expected

    def test_rubric_hallucination_cannot_be_zero(self, client: TestClient):
        """AC3: Hallucination weight of 0 returns 422."""
        rubric = {**VALID_RUBRIC, "hallucination": 0, "accuracy": 50}
        resp = client.post("/rubric/validate", json=rubric)
        assert resp.status_code == 422
        detail = resp.json()["detail"][0]["msg"] if isinstance(resp.json()["detail"], list) else resp.json()["detail"]
        assert "hallucination" in detail.lower() or "10" in detail

    def test_rubric_hallucination_minimum_is_10(self, client: TestClient):
        """AC3: Hallucination weight of 9 is rejected (min is 10)."""
        rubric = {**VALID_RUBRIC, "hallucination": 9, "accuracy": 41}
        resp = client.post("/rubric/validate", json=rubric)
        assert resp.status_code == 422

    def test_rubric_hallucination_minimum_10_is_accepted(self, client: TestClient):
        """AC3: Hallucination weight of exactly 10 is accepted."""
        rubric = {"accuracy": 30, "hallucination": 10, "instruction_following": 30, "conciseness": 20, "cost_efficiency": 10}
        resp = client.post("/rubric/validate", json=rubric)
        assert resp.status_code == 200

    def test_rubric_custom_weights_must_sum_to_100(self, client: TestClient):
        """AC4: Weights summing to 85 are rejected with descriptive error."""
        rubric = {"accuracy": 20, "hallucination": 20, "instruction_following": 20, "conciseness": 15, "cost_efficiency": 10}
        resp = client.post("/rubric/validate", json=rubric)
        assert resp.status_code == 422
        detail = str(resp.json())
        assert "100" in detail

    def test_rubric_weights_sum_over_100_rejected(self, client: TestClient):
        """AC4: Weights summing to 115 are also rejected."""
        rubric = {"accuracy": 30, "hallucination": 30, "instruction_following": 30, "conciseness": 15, "cost_efficiency": 10}
        resp = client.post("/rubric/validate", json=rubric)
        assert resp.status_code == 422

    def test_valid_rubric_accepted(self, client: TestClient):
        """AC4: Valid rubric (sums to 100, hallucination >= 10) passes."""
        resp = client.post("/rubric/validate", json=VALID_RUBRIC)
        assert resp.status_code == 200
        body = resp.json()
        assert body["hallucination"] == 25


# ══════════════════════════════════════════════════════════════════════════
# SQLite — Persistence
# ══════════════════════════════════════════════════════════════════════════

class TestDatabase:
    EXPECTED_TABLES = {"eval_runs", "prompts", "model_results", "verdicts"}

    def test_db_tables_created_on_startup(self, setup_test_db):
        """All 4 PRD tables exist after startup."""
        tables = set(get_table_names())
        assert self.EXPECTED_TABLES.issubset(tables), (
            f"Missing tables: {self.EXPECTED_TABLES - tables}"
        )

    def test_eval_run_saved_to_db(self, client: TestClient):
        """Triggering a mock eval run creates a row in eval_runs with status=complete."""
        from backend.db.database import SessionLocal
        from backend.db.models import EvalRun

        resp = client.post("/eval/run", json=valid_run_request())
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        with SessionLocal() as db:
            run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
        assert run is not None
        assert run.status == "complete"
        assert run.id == run_id

    def test_prompts_saved_to_db(self, client: TestClient):
        """Prompts are persisted as individual rows linked to the eval run."""
        from backend.db.database import SessionLocal
        from backend.db.models import Prompt

        n = 7
        resp = client.post("/eval/run", json=valid_run_request(n_prompts=n))
        run_id = resp.json()["run_id"]

        with SessionLocal() as db:
            count = db.query(Prompt).filter(Prompt.eval_run_id == run_id).count()
        assert count == n

    def test_model_results_saved_to_db(self, client: TestClient):
        """Model results are saved: n_prompts × n_models rows in model_results."""
        from backend.db.database import SessionLocal
        from backend.db.models import ModelResult

        n_prompts = 5
        models = ["gpt-5-4", "claude-sonnet-4-6"]
        resp = client.post("/eval/run", json=valid_run_request(n_prompts=n_prompts, models=models))
        run_id = resp.json()["run_id"]

        with SessionLocal() as db:
            count = db.query(ModelResult).filter(ModelResult.eval_run_id == run_id).count()
        assert count == n_prompts * len(models)

    def test_verdict_saved_to_db(self, client: TestClient):
        """A verdict row is created for each completed eval run."""
        from backend.db.database import SessionLocal
        from backend.db.models import Verdict

        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]

        with SessionLocal() as db:
            verdict = db.query(Verdict).filter(Verdict.eval_run_id == run_id).first()
        assert verdict is not None
        assert verdict.winning_model == "gpt-5-4"


# ══════════════════════════════════════════════════════════════════════════
# DEV_MODE — Mock judge
# ══════════════════════════════════════════════════════════════════════════

class TestDevMode:
    def test_dev_mode_returns_mock_scores(self, client: TestClient):
        """DEV_MODE=true returns scores that exactly match models.yaml mock_scores."""
        resp = client.post("/eval/run", json=valid_run_request())
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        results_resp = client.get(f"/eval/{run_id}/results")
        assert results_resp.status_code == 200
        results = results_resp.json()["results"]
        assert len(results) > 0

        scores = results[0]["dimension_scores"]
        assert scores["accuracy"] == MOCK_SCORES["accuracy"]
        assert scores["hallucination"] == MOCK_SCORES["hallucination"]
        assert scores["instruction_following"] == MOCK_SCORES["instruction_following"]
        assert scores["conciseness"] == MOCK_SCORES["conciseness"]
        # Session 4: cost_efficiency is no longer stored per-result in dimension_scores.
        # It is calculated per-model across all prompts and stored in verdict.score_breakdown.
        # Verify it is NOT in dimension_scores (correct design post-Session-4).
        assert "cost_efficiency" not in scores, (
            "cost_efficiency must not be stored in dimension_scores (it lives in verdict.score_breakdown)"
        )

    def test_dev_mode_never_calls_real_apis(self, client: TestClient, mocker):
        """DEV_MODE=true must not invoke any real provider API client."""
        mock_openai = mocker.patch("openai.AsyncOpenAI", side_effect=AssertionError("OpenAI called in DEV_MODE"))
        mock_anthropic = mocker.patch("anthropic.AsyncAnthropic", side_effect=AssertionError("Anthropic called in DEV_MODE"))

        resp = client.post("/eval/run", json=valid_run_request())
        assert resp.status_code == 200  # would have raised if mock was called

    def test_dev_mode_mock_response_text_present(self, client: TestClient):
        """Mock responses contain expected [MOCK] prefix from models.yaml."""
        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]
        results_resp = client.get(f"/eval/{run_id}/results")
        result = results_resp.json()["results"][0]
        assert "[MOCK]" in result["response_text"]

    def test_dev_mode_reasoning_present_for_all_dimensions(self, client: TestClient):
        """Every judge-scored dimension has reasoning text attached."""
        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]
        results_resp = client.get(f"/eval/{run_id}/results")
        result = results_resp.json()["results"][0]
        for dim in ["accuracy", "hallucination", "instruction_following", "conciseness"]:
            assert dim in result["dimension_reasoning"]
            assert len(result["dimension_reasoning"][dim]) > 0

    def test_results_endpoint_returns_404_for_unknown_run(self, client: TestClient):
        """Unknown run_id returns 404."""
        resp = client.get("/eval/nonexistent-run-id/results")
        assert resp.status_code == 404
