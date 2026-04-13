import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text

from backend.db.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)
    modality = Column(String, nullable=False)
    rubric_config = Column(JSON, nullable=False)
    models_selected = Column(JSON, nullable=False)
    engineer_name = Column(String, nullable=True)   # per-run override (Run tab field)
    engineer_names = Column(JSON, nullable=True)    # JSON list — unique names from all prompts
    status = Column(String, nullable=False, default="pending")
    progress_pct = Column(Float, nullable=True, default=0.0)  # 0-100, updated during runner
    custom_label = Column(String, nullable=True)
    error_message = Column(String, nullable=True)   # set on status=failed
    completed_at = Column(DateTime, nullable=True)  # set on status=complete or failed
    # NOTE: API keys are NEVER stored — only kept in request memory


class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(String, primary_key=True, default=_uuid)
    eval_run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    prompt_text = Column(String, nullable=False)
    image_path = Column(String, nullable=True)
    image_data = Column(Text, nullable=True)        # base64 data URI: "data:image/jpeg;base64,..."
    expected_output = Column(String, nullable=True)
    engineer_name = Column(String, nullable=True)


class ModelResult(Base):
    __tablename__ = "model_results"

    id = Column(String, primary_key=True, default=_uuid)
    eval_run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    prompt_id = Column(String, ForeignKey("prompts.id"), nullable=False)
    prompt_index = Column(String, nullable=False, default="0")
    model_name = Column(String, nullable=False)
    response_text = Column(String, nullable=False)
    dimension_scores = Column(JSON, nullable=False)
    dimension_reasoning = Column(JSON, nullable=False)
    hallucination_flagged = Column(Boolean, nullable=False, default=False)
    hallucination_reason = Column(String, nullable=True)
    tokens_used = Column(JSON, nullable=False)      # {"input": N, "output": N} — kept for compat
    tokens_in = Column(Integer, nullable=True)      # input tokens (separate column for queries)
    tokens_out = Column(Integer, nullable=True)     # output tokens
    cost_usd = Column(Float, nullable=False, default=0.0)


class Verdict(Base):
    __tablename__ = "verdicts"

    id = Column(String, primary_key=True, default=_uuid)
    eval_run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    winning_model = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    score_breakdown = Column(JSON, nullable=False)
    cost_comparison = Column(JSON, nullable=False)
    hallucination_warnings = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
