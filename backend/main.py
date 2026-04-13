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
    from backend.db.database import create_tables
    create_tables()
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
