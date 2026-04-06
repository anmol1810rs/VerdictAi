import os
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="VerdictAI", version="0.1.0")

DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"


@app.get("/health")
def health_check():
    return {"status": "ok", "mode": "dev" if DEV_MODE else "prod"}
