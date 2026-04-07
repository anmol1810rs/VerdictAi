"""
Layer 1 — Smoke tests.

Run: pytest backend/tests/test_smoke.py -v
"""
import pytest
from fastapi.testclient import TestClient

from backend.config import DEV_MODE, MODELS_CONFIG, PRICING_CONFIG, MVP_MODEL_IDS
from backend.db.database import get_table_names


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self, client: TestClient):
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_returns_mode(self, client: TestClient):
        response = client.get("/health")
        assert response.json()["mode"] in ("dev", "prod")

    def test_health_mode_is_dev_in_test_env(self, client: TestClient):
        # DEV_MODE=true set in conftest.py
        response = client.get("/health")
        assert response.json()["mode"] == "dev"


class TestConfigLoads:
    def test_models_yaml_loads(self):
        assert MODELS_CONFIG is not None
        assert "mvp_models" in MODELS_CONFIG

    def test_pricing_yaml_loads(self):
        assert PRICING_CONFIG is not None
        assert "models" in PRICING_CONFIG

    def test_six_mvp_models_defined(self):
        assert len(MVP_MODEL_IDS) == 6

    def test_dev_mode_flag_readable(self):
        # Set to true by conftest.py
        assert DEV_MODE is True


class TestDatabaseStartup:
    EXPECTED_TABLES = {"eval_runs", "prompts", "model_results", "verdicts"}

    def test_all_four_tables_exist(self, setup_test_db):
        tables = set(get_table_names())
        assert self.EXPECTED_TABLES.issubset(tables), (
            f"Missing tables: {self.EXPECTED_TABLES - tables}"
        )


class TestUploadEndpointSmoke:
    def test_upload_endpoint_exists(self, client: TestClient):
        """POST /upload without a file returns 422 (not 404 or 405)."""
        response = client.post("/upload")
        assert response.status_code == 422  # missing file — endpoint exists

    def test_rubric_validate_endpoint_exists(self, client: TestClient):
        response = client.post("/rubric/validate", json={})
        assert response.status_code == 422  # missing fields — endpoint exists

    def test_keys_validate_endpoint_exists(self, client: TestClient):
        response = client.post("/keys/validate", json={})
        assert response.status_code == 422  # missing fields — endpoint exists

    def test_eval_run_endpoint_exists(self, client: TestClient):
        response = client.post("/eval/run", json={})
        assert response.status_code == 422  # missing fields — endpoint exists
