from typing import Optional
from pydantic import BaseModel, field_validator, model_validator


# ── Upload ─────────────────────────────────────────────────────────────────

class PromptInput(BaseModel):
    prompt: str
    expected_output: Optional[str] = None
    engineer_name: Optional[str] = None


class ValidationWarning(BaseModel):
    field: str
    message: str


class UploadResponse(BaseModel):
    prompt_count: int
    has_ground_truth: bool
    has_engineer_names: bool
    modality: str
    prompts: list[PromptInput]
    warnings: list[ValidationWarning] = []
    validation_summary: Optional[str] = None


# ── API Keys ───────────────────────────────────────────────────────────────

class APIKeys(BaseModel):
    openai_api_key: str
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    @field_validator("openai_api_key")
    @classmethod
    def openai_key_format(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("sk-") or len(v) < 20:
            raise ValueError(
                "OpenAI API key must start with 'sk-' and be at least 20 characters. "
                "Check your key at platform.openai.com."
            )
        return v


class KeyValidationResponse(BaseModel):
    valid: bool
    openai: bool
    anthropic: bool
    google: bool


# ── Rubric ─────────────────────────────────────────────────────────────────

class RubricWeights(BaseModel):
    accuracy: int
    hallucination: int
    instruction_following: int
    conciseness: int
    cost_efficiency: int

    @field_validator("hallucination")
    @classmethod
    def hallucination_min_weight(cls, v: int) -> int:
        if v < 10:
            raise ValueError(
                "Hallucination weight cannot be less than 10. "
                "This is mandatory to ensure hallucination is always evaluated."
            )
        return v

    @model_validator(mode="after")
    def weights_sum_to_100(self) -> "RubricWeights":
        total = (
            self.accuracy
            + self.hallucination
            + self.instruction_following
            + self.conciseness
            + self.cost_efficiency
        )
        if total != 100:
            raise ValueError(
                f"Rubric weights must sum to 100. Current total: {total}. "
                f"Adjust dimensions so they add up to exactly 100."
            )
        return self


# ── Eval run ───────────────────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    prompts: list[PromptInput]
    models_selected: list[str]
    rubric: RubricWeights
    api_keys: APIKeys
    engineer_name: Optional[str] = None
    custom_label: Optional[str] = None


class DimensionScores(BaseModel):
    accuracy: float
    hallucination: float
    instruction_following: float
    conciseness: float
    cost_efficiency: float


class ModelResultOut(BaseModel):
    model_name: str
    prompt_index: int
    response_text: str
    dimension_scores: dict
    dimension_reasoning: dict
    hallucination_flagged: bool
    tokens_used: dict
    cost_usd: float


class EvalRunResponse(BaseModel):
    run_id: str
    status: str


class EvalResultsResponse(BaseModel):
    run_id: str
    status: str
    results: list[ModelResultOut]
    verdict: Optional[dict] = None


# ── Story 1.7 — Status polling ─────────────────────────────────────────────

class EvalStatusResponse(BaseModel):
    run_id: str
    status: str                          # pending | running | complete | failed
    error_message: Optional[str] = None
    completed_at: Optional[str] = None  # ISO string, set on complete/failed


# ── Story 1.8 — Run history ────────────────────────────────────────────────

class EvalHistoryItem(BaseModel):
    id: str
    created_at: str                      # formatted: "07 Apr 2026, 8:00pm"
    modality: str
    models_selected: list[str]
    engineer_names: list[str]
    run_label: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    winning_model: Optional[str] = None
    overall_score: Optional[float] = None  # full scoring in Session 4


class EvalHistoryResponse(BaseModel):
    runs: list[EvalHistoryItem]
    total: int


# ── Modality & Models ───────────────────────────────────────────────────────

class IncompatibleModel(BaseModel):
    model: str
    reason: str


class ModelsCompatibleResponse(BaseModel):
    modality: str
    compatible_models: list[dict]
    incompatible_models: list[IncompatibleModel]
    suggestions: dict
