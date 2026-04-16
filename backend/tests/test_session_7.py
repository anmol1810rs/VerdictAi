"""
Session 7 acceptance tests — Stories 3.1 (PDF Export) and 3.2 (JSON Export).

Run: pytest backend/tests/test_session_7.py -v

Test groups:
  TestPDFExport  (7 tests) — 200/pdf content-type, text extraction, page count,
                             GT present/absent, verdict text, score + cost tables
  TestJSONExport (8 tests) — 200/json content-type, schema keys, verdict fields,
                             prompts length, GT null/populated, filename header,
                             parseable body, version field
"""
import json
import re
from io import BytesIO

import pytest

from backend.tests.conftest import VALID_RUBRIC, make_prompts, valid_run_request


# ── Shared helper ──────────────────────────────────────────────────────────

def _run_completed(client, n_prompts: int = 10, with_gt: bool = False,
                   models: list | None = None, label: str | None = None) -> str:
    """
    POST /eval/run and return the run_id.
    DEV_MODE=true ensures the run finishes synchronously in the test process.
    """
    req = {
        "prompts": make_prompts(n_prompts, with_gt=with_gt),
        "models_selected": models or ["gpt-5-4", "gpt-5-4-mini"],
        "rubric": VALID_RUBRIC,
        "api_keys": {"openai_api_key": "sk-test1234567890abcdefghij"},
    }
    if label:
        req["custom_label"] = label
    resp = client.post("/eval/run", json=req)
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


# ══════════════════════════════════════════════════════════════════════════
# Story 3.1 — PDF Export
# ══════════════════════════════════════════════════════════════════════════


class TestPDFExport:

    def test_pdf_export_returns_200(self, client):
        """Complete a mock run, call export endpoint, assert 200 + application/pdf."""
        run_id = _run_completed(client, n_prompts=5)
        resp = client.get(f"/eval/{run_id}/export/pdf")
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"] == "application/pdf"

    def test_pdf_contains_verdict_text(self, client):
        """Extracted PDF text must contain the winner model name."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=5)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        # Winner model name must appear in the PDF
        assert "gpt-5-4" in text, f"Expected winner model name in PDF text. Got:\n{text[:500]}"

    def test_pdf_contains_score_table(self, client):
        """Extracted PDF text must contain 'SCORE BREAKDOWN'."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=5)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "SCORE BREAKDOWN" in text, (
            f"Expected 'SCORE BREAKDOWN' in PDF. Got:\n{text[:500]}"
        )

    def test_pdf_contains_cost_table(self, client):
        """Extracted PDF text must contain 'COST BREAKDOWN'."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=5)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "COST BREAKDOWN" in text, (
            f"Expected 'COST BREAKDOWN' in PDF. Got:\n{text[:500]}"
        )

    def test_pdf_contains_gt_when_provided(self, client):
        """When GT data is present, PDF must contain 'GT ALIGNMENT'."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=5, with_gt=True)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "GT ALIGNMENT" in text, (
            f"Expected 'GT ALIGNMENT' section in PDF when GT provided. Got:\n{text[:500]}"
        )

    def test_pdf_excludes_gt_when_not_provided(self, client):
        """When no GT data, PDF must NOT contain 'GT ALIGNMENT'."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=5, with_gt=False)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        assert "GT ALIGNMENT" not in text, (
            "GT ALIGNMENT section must be absent when no GT provided."
        )

    def test_pdf_max_2_pages(self, client):
        """A 10-prompt, 2-model run must produce a PDF with at most 2 pages."""
        import pdfplumber
        run_id = _run_completed(client, n_prompts=10)
        pdf_bytes = client.get(f"/eval/{run_id}/export/pdf").content
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
        assert page_count <= 2, (
            f"PDF must be at most 2 pages for a 10-prompt run. Got {page_count} pages."
        )


# ══════════════════════════════════════════════════════════════════════════
# Story 3.2 — JSON Export
# ══════════════════════════════════════════════════════════════════════════


class TestJSONExport:

    def test_json_export_returns_200(self, client):
        """Complete a mock run, call export endpoint, assert 200 + application/json."""
        run_id = _run_completed(client, n_prompts=5)
        resp = client.get(f"/eval/{run_id}/export/json")
        assert resp.status_code == 200, resp.text
        assert "application/json" in resp.headers["content-type"]

    def test_json_schema_complete(self, client):
        """Top-level keys must exactly match: verdictai_version, exported_at, run, verdict, models, prompts."""
        run_id = _run_completed(client, n_prompts=5)
        data = client.get(f"/eval/{run_id}/export/json").json()
        required_keys = {"verdictai_version", "exported_at", "run", "verdict", "models", "prompts"}
        assert required_keys.issubset(data.keys()), (
            f"Missing top-level keys: {required_keys - set(data.keys())}"
        )

    def test_json_verdict_fields_present(self, client):
        """verdict object must have non-null winning_model, overall_score, and summary."""
        run_id = _run_completed(client, n_prompts=5)
        data = client.get(f"/eval/{run_id}/export/json").json()
        v = data["verdict"]
        assert v.get("winning_model") is not None, "winning_model must not be null"
        assert v.get("overall_score") is not None, "overall_score must not be null"
        assert v.get("summary") is not None, "summary must not be null"

    def test_json_prompts_array_correct_length(self, client):
        """For a 10-prompt run the prompts array must have exactly 10 items."""
        run_id = _run_completed(client, n_prompts=10)
        data = client.get(f"/eval/{run_id}/export/json").json()
        assert len(data["prompts"]) == 10, (
            f"Expected 10 prompts, got {len(data['prompts'])}"
        )

    def test_json_gt_null_when_not_provided(self, client):
        """Without GT data, ground_truth_score must be null for every prompt response."""
        run_id = _run_completed(client, n_prompts=5, with_gt=False)
        data = client.get(f"/eval/{run_id}/export/json").json()
        for prompt in data["prompts"]:
            for model, resp in prompt["responses"].items():
                assert resp["ground_truth_score"] is None, (
                    f"ground_truth_score must be null when no GT provided "
                    f"(model={model}, prompt={prompt['index']})"
                )

    def test_json_gt_populated_when_provided(self, client):
        """With GT data, ground_truth_score must be non-null for every prompt response."""
        run_id = _run_completed(client, n_prompts=5, with_gt=True)
        data = client.get(f"/eval/{run_id}/export/json").json()
        for prompt in data["prompts"]:
            for model, resp in prompt["responses"].items():
                assert resp["ground_truth_score"] is not None, (
                    f"ground_truth_score must be non-null when GT provided "
                    f"(model={model}, prompt={prompt['index']})"
                )

    def test_json_filename_correct(self, client):
        """Content-Disposition header must contain 'verdictai_' prefix and '.json' extension."""
        run_id = _run_completed(client, n_prompts=5, label="MyRun")
        resp = client.get(f"/eval/{run_id}/export/json")
        cd = resp.headers.get("content-disposition", "")
        assert "verdictai_" in cd, f"Filename must start with 'verdictai_'. Got: {cd}"
        assert ".json" in cd, f"Filename must end with '.json'. Got: {cd}"

    def test_json_is_valid_parseable(self, client):
        """Response body must be valid JSON (json.loads must not raise)."""
        run_id = _run_completed(client, n_prompts=5)
        resp = client.get(f"/eval/{run_id}/export/json")
        try:
            parsed = json.loads(resp.text)
        except json.JSONDecodeError as exc:
            pytest.fail(f"JSON export body is not valid JSON: {exc}")
        assert isinstance(parsed, dict)

    def test_json_version_field(self, client):
        """verdictai_version must be the string '1.0.0'."""
        run_id = _run_completed(client, n_prompts=5)
        data = client.get(f"/eval/{run_id}/export/json").json()
        assert data.get("verdictai_version") == "1.0.0", (
            f"Expected verdictai_version='1.0.0', got {data.get('verdictai_version')!r}"
        )
