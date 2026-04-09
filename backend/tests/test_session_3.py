"""
Layer 2 — Session 3 story acceptance tests.

Stories 1.7 (Eval Run Persistence) and 1.8 (Run History Sidebar).

Run: pytest backend/tests/test_session_3.py -v
"""
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base, get_db
from backend.db.models import EvalRun, Verdict
from backend.main import app
from backend.tests.conftest import (
    VALID_API_KEYS,
    VALID_RUBRIC,
    make_prompts,
    valid_run_request,
)


# ══════════════════════════════════════════════════════════════════════════
# Story 1.7 — Eval Run Persistence
# ══════════════════════════════════════════════════════════════════════════

class TestEvalRunPersistence:

    def test_eval_run_created_immediately_on_start(self, client):
        """
        Run row must exist in SQLite with status=pending immediately on start,
        before the background eval task updates it.
        Patch _execute_eval to a no-op so we can observe the pending state.
        """
        with mock.patch("backend.eval.router._execute_eval"):
            resp = client.post("/eval/run", json=valid_run_request())
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]

        # Verify pending row exists in DB
        from backend.db.database import SessionLocal
        db = SessionLocal()
        try:
            run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
            assert run is not None, "Run row must exist immediately after POST"
            assert run.status == "pending", f"Expected pending, got {run.status}"
        finally:
            db.close()

    def test_status_updates_to_running(self, client):
        """
        Background task sets status=running before executing model calls.
        Simulate this by patching _execute_eval to only set running, not complete.
        """
        def only_set_running(run_id, request):
            from backend.db.database import SessionLocal
            db = SessionLocal()
            try:
                run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
                if run:
                    run.status = "running"
                    db.commit()
            finally:
                db.close()

        with mock.patch("backend.eval.router._execute_eval", side_effect=only_set_running):
            resp = client.post("/eval/run", json=valid_run_request())
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]

        status_resp = client.get(f"/eval/{run_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "running"

    def test_status_updates_to_complete(self, client):
        """
        After a successful eval run, status must be complete and completed_at set.
        """
        resp = client.post("/eval/run", json=valid_run_request())
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        status_resp = client.get(f"/eval/{run_id}/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "complete"
        assert data["completed_at"] is not None, "completed_at must be set on complete"
        assert data["error_message"] is None

    def test_status_updates_to_failed(self, client):
        """
        If background eval raises an exception, status=failed and error_message saved.
        """
        def always_fail(run_id, request):
            from backend.db.database import SessionLocal
            db = SessionLocal()
            try:
                run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
                if run:
                    run.status = "running"
                    db.commit()
                    run.status = "failed"
                    run.error_message = "Simulated API timeout"
                    db.commit()
            finally:
                db.close()

        with mock.patch("backend.eval.router._execute_eval", side_effect=always_fail):
            resp = client.post("/eval/run", json=valid_run_request())
            assert resp.status_code == 200
            run_id = resp.json()["run_id"]

        status_resp = client.get(f"/eval/{run_id}/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "Simulated API timeout"

    def test_page_refresh_restores_correct_status(self, client):
        """
        Simulate page refresh by re-fetching GET /eval/{id}/status after a completed run.
        Status must match DB state — not reset to pending.
        """
        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]

        # First poll
        first = client.get(f"/eval/{run_id}/status").json()
        assert first["status"] == "complete"

        # Second poll (simulated page refresh)
        second = client.get(f"/eval/{run_id}/status").json()
        assert second["status"] == "complete", "Status must persist across re-fetches"
        assert second["run_id"] == run_id

    def test_run_label_saved_to_eval_runs(self, client):
        """
        custom_label submitted with run must be persisted to eval_runs table.
        """
        req = valid_run_request()
        req["custom_label"] = "Test batch 1"

        resp = client.post("/eval/run", json=req)
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        from backend.db.database import SessionLocal
        db = SessionLocal()
        try:
            run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
            assert run is not None
            assert run.custom_label == "Test batch 1"
        finally:
            db.close()

    def test_auto_save_requires_no_user_action(self, client):
        """
        Run must exist in DB without any explicit save call — just POST /eval/run.
        """
        req = valid_run_request()
        resp = client.post("/eval/run", json=req)
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        from backend.db.database import SessionLocal
        db = SessionLocal()
        try:
            run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
            assert run is not None, "Run must exist in DB without any explicit save"
            assert run.rubric_config is not None
            assert run.models_selected is not None
        finally:
            db.close()


# ══════════════════════════════════════════════════════════════════════════
# Story 1.8 — Run History Sidebar
# ══════════════════════════════════════════════════════════════════════════

class TestRunHistory:

    def test_empty_state_when_no_runs(self):
        """
        GET /eval/history on a fresh DB must return empty list (not an error).
        Uses an isolated in-memory DB to guarantee no pre-existing runs.
        """
        # Create a fresh in-memory SQLite DB.
        # StaticPool is required so all connections share the same in-memory DB;
        # without it each new connection creates a separate empty database.
        from sqlalchemy.pool import StaticPool
        test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        from backend.db import models as _  # noqa: F401 — register models with Base
        Base.metadata.create_all(bind=test_engine)
        TestSession = sessionmaker(bind=test_engine)

        def override_get_db():
            db = TestSession()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        fresh_client = TestClient(app)

        try:
            resp = fresh_client.get("/eval/history")
            assert resp.status_code == 200
            data = resp.json()
            assert data["runs"] == []
            assert data["total"] == 0
        finally:
            app.dependency_overrides.clear()

    def test_history_returns_all_runs_ordered_by_recency(self, client):
        """
        GET /eval/history must return runs ordered most-recent first.
        Create 3 runs and assert ordering by checking created_at descending.
        """
        # Create 3 runs
        run_ids = []
        for _ in range(3):
            resp = client.post("/eval/run", json=valid_run_request())
            assert resp.status_code == 200
            run_ids.append(resp.json()["run_id"])

        hist_resp = client.get("/eval/history")
        assert hist_resp.status_code == 200
        data = hist_resp.json()
        assert data["total"] >= 3

        # All 3 created runs must appear in history
        history_ids = [r["id"] for r in data["runs"]]
        for run_id in run_ids:
            assert run_id in history_ids, f"Run {run_id} missing from history"

        # Verify descending order: each run's created_at must be >= the next one
        # We can't directly compare formatted strings, but we can verify our 3 runs
        # appear with the last-created first
        positions = {run_id: history_ids.index(run_id) for run_id in run_ids}
        # run_ids[-1] was created last, must have the smallest index (most recent)
        assert positions[run_ids[-1]] < positions[run_ids[0]], \
            "Most recently created run must appear before older runs"

    def test_history_filter_by_model(self, client):
        """
        GET /eval/history?model=... must return only runs that used that model.
        """
        # Create a run with gemini-2-5-flash (unique model for this test)
        req_flash = valid_run_request(models=["gemini-2-5-flash"])
        resp_flash = client.post("/eval/run", json=req_flash)
        assert resp_flash.status_code == 200
        flash_run_id = resp_flash.json()["run_id"]

        # Create a run with gpt-5-4
        req_gpt = valid_run_request(models=["gpt-5-4"])
        resp_gpt = client.post("/eval/run", json=req_gpt)
        gpt_run_id = resp_gpt.json()["run_id"]

        # Filter by gemini-2-5-flash
        hist_resp = client.get("/eval/history", params={"model": "gemini-2-5-flash"})
        assert hist_resp.status_code == 200
        data = hist_resp.json()
        filtered_ids = [r["id"] for r in data["runs"]]

        assert flash_run_id in filtered_ids, "gemini-2-5-flash run must appear in filtered results"
        assert gpt_run_id not in filtered_ids, "gpt-5-4-only run must NOT appear in filtered results"

    def test_history_filter_by_engineer(self, client):
        """
        GET /eval/history?engineer=... must return only runs with matching engineer names.
        """
        # Create run with engineer name "Zara" in prompts
        req = valid_run_request()
        for p in req["prompts"]:
            p["engineer_name"] = "Zara"
        resp = client.post("/eval/run", json=req)
        assert resp.status_code == 200
        zara_run_id = resp.json()["run_id"]

        # Create run with no engineer names
        req2 = valid_run_request()
        resp2 = client.post("/eval/run", json=req2)
        other_run_id = resp2.json()["run_id"]

        # Filter by engineer=Zara
        hist_resp = client.get("/eval/history", params={"engineer": "Zara"})
        assert hist_resp.status_code == 200
        data = hist_resp.json()
        filtered_ids = [r["id"] for r in data["runs"]]

        assert zara_run_id in filtered_ids, "Zara's run must appear in filtered results"
        assert other_run_id not in filtered_ids, "Run without engineer must NOT appear in Zara filter"

    def test_clicking_run_loads_full_results(self, client):
        """
        GET /eval/{id}/results for a completed run must return all required fields.
        This is the API that 'clicking' a history entry calls.
        """
        resp = client.post("/eval/run", json=valid_run_request())
        run_id = resp.json()["run_id"]

        results_resp = client.get(f"/eval/{run_id}/results")
        assert results_resp.status_code == 200
        data = results_resp.json()

        # All required fields must be present
        assert data["run_id"] == run_id
        assert data["status"] == "complete"
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0, "Results must contain model outputs"
        assert data["verdict"] is not None, "Verdict must be present for completed run"
        assert "winning_model" in data["verdict"]
        assert "summary" in data["verdict"]

    def test_failed_run_shows_error_in_history(self, client):
        """
        A failed run must show status=failed and error_message in GET /eval/history.
        """
        error_text = "Mock failure for history test"

        def always_fail(run_id, request):
            from backend.db.database import SessionLocal
            db = SessionLocal()
            try:
                run = db.query(EvalRun).filter(EvalRun.id == run_id).first()
                if run:
                    run.status = "failed"
                    run.error_message = error_text
                    db.commit()
            finally:
                db.close()

        with mock.patch("backend.eval.router._execute_eval", side_effect=always_fail):
            resp = client.post("/eval/run", json=valid_run_request())
            assert resp.status_code == 200
            failed_run_id = resp.json()["run_id"]

        hist_resp = client.get("/eval/history")
        assert hist_resp.status_code == 200
        history_runs = hist_resp.json()["runs"]

        failed_entry = next((r for r in history_runs if r["id"] == failed_run_id), None)
        assert failed_entry is not None, "Failed run must appear in history"
        assert failed_entry["status"] == "failed"
        assert failed_entry["error_message"] == error_text
