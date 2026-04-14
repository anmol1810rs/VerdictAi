import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./verdictai.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
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
    Add columns introduced after the initial table creation.
    Safe to run on existing DBs — catches the 'duplicate column' error and continues.
    """
    from sqlalchemy import text
    stmts = [
        "ALTER TABLE model_results ADD COLUMN variance_score REAL",
    ]
    with engine.connect() as conn:
        for stmt in stmts:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:  # noqa: BLE001 — column already exists
                pass


def create_tables() -> None:
    """Create all tables. Import models first so they register with Base."""
    from backend.db import models as _  # noqa: F401
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns()


def get_table_names() -> list[str]:
    return inspect(engine).get_table_names()
