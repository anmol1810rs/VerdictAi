import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./verdictai.db")

if DATABASE_URL == "sqlite:///:memory:":
    # In-memory SQLite: StaticPool forces all connections to share the same
    # in-memory DB so tables created at setup are visible during tests.
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
elif DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _migrate_add_columns() -> None:
    """
    Idempotent column migrations for schema changes added after initial release.
    Uses IF NOT EXISTS (supported on PostgreSQL 9.6+ and SQLite 3.37+).
    """
    from sqlalchemy import text
    stmts = [
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS variance_score REAL",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS ground_truth_score REAL",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS ground_truth_reasoning TEXT",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS rouge_1_score REAL",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS rouge_l_score REAL",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS evidence_data JSON",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS model_error TEXT",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS eval_api_calls INTEGER DEFAULT 0",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS judge_api_calls INTEGER DEFAULT 0",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS gt_api_calls INTEGER DEFAULT 0",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS judge_tokens_in INTEGER",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS judge_tokens_out INTEGER",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS judge_cost_usd REAL",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS gt_tokens_in INTEGER",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS gt_tokens_out INTEGER",
        "ALTER TABLE model_results ADD COLUMN IF NOT EXISTS gt_cost_usd REAL",
    ]
    with engine.connect() as conn:
        for stmt in stmts:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:  # noqa: BLE001 — column already exists (fallback for old SQLite)
                conn.rollback()


def create_tables() -> None:
    """Create all tables. Import models first so they register with Base."""
    from backend.db import models as _  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns()


def get_table_names() -> list[str]:
    return inspect(engine).get_table_names()
