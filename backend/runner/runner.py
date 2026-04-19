"""
backend/runner/runner.py — Multi-model async runner (Session 4, Story 2.1/2.2)

Executes all selected models against all prompts simultaneously using asyncio.gather().
No sequential execution — all (model × prompt) tasks fire at once.

DEV_MODE=true:  Mock responses from models.yaml. No API calls. Zero cost.
DEV_MODE=false: Real provider API calls (OpenAI / Anthropic / Google).

Each provider uses the client library and request format from models.yaml api_clients section.
Token field paths read from models.yaml token_fields per model.
Costs calculated from pricing.yaml input_per_1m_usd / output_per_1m_usd.
"""
import asyncio
import base64
from typing import Optional

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
try:
    from google import genai as _google_genai
except Exception:  # noqa: BLE001 — optional dependency
    _google_genai = None  # type: ignore[assignment]

import logging

from backend.config import DEV_MODE, DEV_MOCK, MOCK_LATENCY_MS, MODELS_CONFIG, PRICING_CONFIG

logger = logging.getLogger(__name__)

# Build lookup maps at module load time (avoid repeated iteration in hot path)
MODEL_MAP: dict = {m["id"]: m for m in MODELS_CONFIG["mvp_models"]}
PRICING_MAP: dict = PRICING_CONFIG["models"]

JUDGE_DIMENSIONS = ["accuracy", "hallucination", "instruction_following", "conciseness"]


# ── Cost calculation ───────────────────────────────────────────────────────


def calculate_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """
    Calculate cost in USD from pricing.yaml rates.
    Public — used directly by tests.
    """
    pricing = PRICING_MAP.get(model_id, {})
    input_rate = pricing.get("input_per_1m_usd", 0.0)
    output_rate = pricing.get("output_per_1m_usd", 0.0)
    return round((tokens_in * input_rate + tokens_out * output_rate) / 1_000_000, 8)


# ── Provider API calls ─────────────────────────────────────────────────────


async def _call_openai(
    model_config: dict, prompt_text: str, image_data: Optional[str], api_key: str
) -> tuple[str, int, int]:
    """
    Call OpenAI via the Responses API (supports both text and image_text).
    image_data: "data:image/jpeg;base64,<b64>" or None.
    Returns (response_text, tokens_in, tokens_out).
    """
    client = AsyncOpenAI(api_key=api_key)

    content: list = [{"type": "input_text", "text": prompt_text}]

    if image_data:
        # Responses API uses image_url for both remote URLs and base64 data URIs
        content.append({"type": "input_image", "image_url": image_data})

    response = await client.responses.create(
        model=model_config["api_model_string"],
        input=[{"role": "user", "content": content}],
        max_output_tokens=1000,
    )

    text = response.output[0].content[0].text
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return text, tokens_in, tokens_out


async def _call_anthropic(
    model_config: dict, prompt_text: str, image_data: Optional[str], api_key: str
) -> tuple[str, int, int]:
    """
    Call Anthropic messages API.
    image_format: base64_block → {"type": "image", "source": {"type": "base64", ...}}
    """
    client = AsyncAnthropic(api_key=api_key)

    if image_data and "," in image_data:
        # Parse data URI: "data:image/jpeg;base64,<b64>"
        header, b64 = image_data.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
        ]
    else:
        content = prompt_text  # type: ignore[assignment]

    response = await client.messages.create(
        model=model_config["api_model_string"],
        max_tokens=1000,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return text, tokens_in, tokens_out


async def _call_google(
    model_config: dict, prompt_text: str, image_data: Optional[str], api_key: str
) -> tuple[str, int, int]:
    """
    Call Google Gemini via google-genai SDK (replaces deprecated google-generativeai).
    image_format: inline bytes → types.Part.from_bytes(data=..., mime_type=...)
    """
    if _google_genai is None:
        raise ImportError("google-genai is not installed")

    from google.genai import types as _gtypes

    client = _google_genai.Client(api_key=api_key)

    if image_data and "," in image_data:
        header, b64 = image_data.split(",", 1)
        mime = header.split(":")[1].split(";")[0]
        image_bytes = base64.b64decode(b64)
        contents = [
            _gtypes.Part.from_text(text=prompt_text),
            _gtypes.Part.from_bytes(data=image_bytes, mime_type=mime),
        ]
    else:
        contents = prompt_text

    response = await client.aio.models.generate_content(
        model=model_config["api_model_string"],
        contents=contents,
    )

    text = response.text
    tokens_in = response.usage_metadata.prompt_token_count
    tokens_out = response.usage_metadata.candidates_token_count
    return text, tokens_in, tokens_out


async def call_model(
    model_id: str,
    prompt_text: str,
    image_data: Optional[str],
    api_keys: dict,
) -> tuple[str, int, int]:
    """
    Dispatch to the correct provider client.
    Public — used directly by tests that mock at this level.
    Returns (response_text, tokens_in, tokens_out).
    """
    model_config = MODEL_MAP[model_id]
    provider = model_config["provider"]

    if provider == "openai":
        return await _call_openai(
            model_config, prompt_text, image_data, api_keys.get("openai_api_key", "")
        )
    elif provider == "anthropic":
        return await _call_anthropic(
            model_config, prompt_text, image_data, api_keys.get("anthropic_api_key", "")
        )
    elif provider == "google":
        return await _call_google(
            model_config, prompt_text, image_data, api_keys.get("google_api_key", "")
        )
    else:
        raise ValueError(f"Unknown provider for model '{model_id}': {provider}")


# ── Single task ────────────────────────────────────────────────────────────


async def _run_single(
    model_id: str,
    prompt_text: str,
    image_data: Optional[str],
    api_keys: dict,
) -> tuple[str, int, int]:
    """
    Run one model against one prompt.
    DEV_MODE → mock response + fixed tokens. No API call.
    Returns (response_text, tokens_in, tokens_out).
    """
    if DEV_MODE:
        if MOCK_LATENCY_MS > 0:
            await asyncio.sleep(MOCK_LATENCY_MS / 1000)
        response_text = DEV_MOCK["mock_models"].get(
            model_id, f"[MOCK] Response from {model_id} for development testing."
        )
        return response_text, 120, 85
    else:
        logger.info("Calling model=%s prompt_preview='%s...'", model_id, prompt_text[:60])
        try:
            result = await call_model(model_id, prompt_text, image_data, api_keys)
            logger.info("Model=%s OK — tokens_in=%d tokens_out=%d", model_id, result[1], result[2])
            return result
        except Exception as exc:
            logger.error("Model=%s FAILED — %s: %s", model_id, type(exc).__name__, exc)
            raise


# ── Parallel runner ────────────────────────────────────────────────────────


async def run_models_parallel(
    model_ids: list[str],
    prompts_data: list[dict],
    api_keys: dict,
) -> list[dict]:
    """
    Execute all (model × prompt) combinations simultaneously via asyncio.gather().

    prompts_data items must have:
        prompt_id: str
        prompt_text: str
        prompt_index: int
        image_data: Optional[str]   — base64 data URI or None
        expected_output: Optional[str]

    Returns list of result dicts (one per model × prompt):
    {
        model_id, prompt_id, prompt_index, prompt_text,
        expected_output, response_text,
        tokens_in, tokens_out, cost_usd,
        error: Optional[str]   — set if this task failed; others still succeed
    }
    """
    tasks = []
    task_meta = []

    for p in prompts_data:
        for model_id in model_ids:
            tasks.append(
                _run_single(model_id, p["prompt_text"], p.get("image_data"), api_keys)
            )
            task_meta.append(
                {
                    "model_id": model_id,
                    "prompt_id": p["prompt_id"],
                    "prompt_index": p["prompt_index"],
                    "prompt_text": p["prompt_text"],
                    "expected_output": p.get("expected_output"),
                }
            )

    # Fire all tasks simultaneously; return_exceptions keeps others running if one fails
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for meta, raw in zip(task_meta, raw_results):
        if isinstance(raw, Exception):
            results.append(
                {
                    **meta,
                    "response_text": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "error": str(raw),
                }
            )
        else:
            resp_text, tin, tout = raw
            cost = (
                round((tin + tout) * 0.000002, 6)  # mock cost in DEV_MODE
                if DEV_MODE
                else calculate_cost(meta["model_id"], tin, tout)
            )
            results.append(
                {
                    **meta,
                    "response_text": resp_text,
                    "tokens_in": tin,
                    "tokens_out": tout,
                    "cost_usd": cost,
                    "error": None,
                }
            )

    return results
