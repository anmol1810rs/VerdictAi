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


def create_tables() -> None:
    """Create all tables. Import models first so they register with Base."""
    from backend.db import models as _  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_table_names() -> list[str]:
    return inspect(engine).get_table_names()
