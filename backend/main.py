import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("verdictai.log", mode="a"),
    ],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup
    from backend.db.database import create_tables, SessionLocal
    create_tables()

    # Recovery: any run left in pending/running state from a previous process is
    # unrecoverable — mark it failed so the UI doesn't show it as forever "running".
    from datetime import datetime, timezone
    db = SessionLocal()
    try:
        from backend.db.models import EvalRun
        stuck = (
            db.query(EvalRun)
            .filter(EvalRun.status.in_(["pending", "running"]))
            .all()
        )
        for run in stuck:
            run.status = "failed"
            run.error_message = "Run interrupted by server restart"
            run.completed_at = datetime.now(timezone.utc)
        if stuck:
            db.commit()
            logging.getLogger(__name__).warning(
                "Startup recovery: marked %d stuck run(s) as failed", len(stuck)
            )
    finally:
        db.close()

    yield


app = FastAPI(
    title="VerdictAI",
    description="Open-source LLM evaluation engine.",
    version="0.1.0",
    lifespan=lifespan,
)

# Register routers
from backend.eval.router import router as eval_router  # noqa: E402
app.include_router(eval_router, tags=["eval"])


@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "mode": "dev" if DEV_MODE else "prod"}
