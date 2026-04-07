"""
Test configuration for VerdictAI.

IMPORTANT: env vars MUST be set before any backend import so that
database.py reads the test DATABASE_URL when the engine is created.
"""
import os

# Override BEFORE any backend imports
os.environ["DATABASE_URL"] = "sqlite:///./test_verdictai.db"
os.environ["DEV_MODE"] = "true"
os.environ["MOCK_LATENCY_MS"] = "0"  # no sleep in tests

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.db.database import Base, create_tables, engine  # noqa: E402
from backend.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create tables once for the whole test session, clean up after."""
    create_tables()
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    try:
        os.remove("./test_verdictai.db")
    except (FileNotFoundError, PermissionError):
        pass


@pytest.fixture
def client(setup_test_db):
    return TestClient(app)


# ── Shared test data ───────────────────────────────────────────────────────

def make_prompts(n: int = 10, with_gt: bool = False, with_engineer: bool = False) -> list[dict]:
    rows = []
    for i in range(n):
        row: dict = {"prompt": f"Test prompt number {i + 1}. Please respond with a short answer."}
        if with_gt:
            row["expected_output"] = f"Expected answer {i + 1}"
        if with_engineer:
            row["engineer_name"] = "Alice"
        rows.append(row)
    return rows


VALID_RUBRIC = {
    "accuracy": 25,
    "hallucination": 25,
    "instruction_following": 25,
    "conciseness": 15,
    "cost_efficiency": 10,
}

VALID_API_KEYS = {
    "openai_api_key": "sk-test1234567890abcdefghij",
}

def valid_run_request(n_prompts: int = 5, models: list | None = None) -> dict:
    return {
        "prompts": make_prompts(n_prompts),
        "models_selected": models or ["gpt-5-4"],
        "rubric": VALID_RUBRIC,
        "api_keys": VALID_API_KEYS,
    }
