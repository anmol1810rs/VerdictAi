import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Core flags ────────────────────────────────────────────
DEV_MODE: bool = os.getenv("DEV_MODE", "true").lower() == "true"
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./verdictai.db")
MOCK_LATENCY_MS: int = int(os.getenv("MOCK_LATENCY_MS", "900"))

# ── YAML config loaders ───────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent


def _load_yaml(filename: str) -> dict:
    with open(REPO_ROOT / filename, "r") as f:
        return yaml.safe_load(f)


MODELS_CONFIG: dict = _load_yaml("models.yaml")
PRICING_CONFIG: dict = _load_yaml("pricing.yaml")

# ── Derived constants ─────────────────────────────────────
MVP_MODEL_IDS: list[str] = [m["id"] for m in MODELS_CONFIG["mvp_models"]]
RUBRIC_PRESETS: dict = MODELS_CONFIG["rubric_presets"]
DEV_MOCK: dict = MODELS_CONFIG["dev_mode"]

MAX_PROMPTS = 100
MIN_PROMPTS = 5

# ── Supabase Auth ──────────────────────────────────────────
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
