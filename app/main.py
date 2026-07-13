"""FastAPI entrypoint for BloomBot.

Exposes the bouquet recommendation chain over HTTP:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time

import openai
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.chain import recommend
from app.logging_config import get_logger

logger = logging.getLogger(__name__)

# Structured JSON logger for per-request observability data.
obs_logger = get_logger()

# Rate limiter keyed on the client's IP address.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="BloomBot",
    description="AI bouquet recommendation API.",
    version="0.1.0",
)

# Wire slowapi into the app: expose the limiter and register the default
# handler that returns a standard 429 response when a limit is exceeded.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The customer's natural-language request.",
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # min_length only counts characters; reject whitespace-only queries too.
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class RecommendMeta(BaseModel):
    """Observability metadata for a single recommendation.

    Exposed to the client so UIs can show timing/token diagnostics. These
    numbers are also emitted to the structured logs via :func:`_log_request`.
    """

    retrieval_time_ms: float = Field(..., description="Time spent in retrieval.")
    llm_time_ms: float = Field(..., description="Time spent in the LLM call.")
    prompt_tokens: int = Field(..., description="Tokens in the prompt.")
    completion_tokens: int = Field(..., description="Tokens in the completion.")
    total_tokens: int = Field(..., description="Prompt + completion tokens.")


class RecommendResponse(BaseModel):
    recommendation: str = Field(..., description="The florist assistant's reply.")
    meta: RecommendMeta = Field(..., description="Timing and token diagnostics.")


@app.get("/")
def health_check() -> dict:
    """Simple liveness check."""
    return {"status": "ok", "service": "BloomBot"}


@app.post("/recommend", response_model=RecommendResponse)
@limiter.limit("10/minute")
def recommend_bouquet(
    request: Request, payload: RecommendRequest
) -> RecommendResponse:
    """Return an AI-generated bouquet recommendation for the customer's query.

    The ``request`` parameter is required by slowapi to read the client IP; the
    request body is bound to ``payload``.
    """
    # Length limits (1-500 chars) are enforced by RecommendRequest validation.
    start = time.perf_counter()
    try:
        result = recommend(payload.query)
    except openai.RateLimitError as exc:
        # Transient upstream capacity issue — ask the caller to retry later.
        # No internal details are exposed in the response.
        logger.warning("OpenAI rate limit hit: %s", exc)
        _log_request(payload.query, start, error=exc)
        raise HTTPException(
            status_code=503, detail="Service temporarily busy, please try again later."
        ) from exc
    except openai.AuthenticationError as exc:
        # Misconfigured/invalid API key — a server-side problem. Log the real
        # cause for operators, but never leak it to the client.
        logger.error("OpenAI authentication failed: %s", exc)
        _log_request(payload.query, start, error=exc)
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — generic fallback
        logger.exception("Recommendation failed: %s", exc)
        _log_request(payload.query, start, error=exc)
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from exc

    _log_request(payload.query, start, result=result)
    return RecommendResponse(
        recommendation=result["recommendation"],
        meta=RecommendMeta(
            retrieval_time_ms=result["retrieval_time_ms"],
            llm_time_ms=result["llm_time_ms"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            total_tokens=result["total_tokens"],
        ),
    )


def _log_request(
    query: str,
    start: float,
    *,
    result: dict | None = None,
    error: Exception | None = None,
) -> None:
    """Emit one structured JSON log line for a /recommend request.

    On success, ``result`` is the dict returned by :func:`recommend`. On
    failure, ``error`` is the raised exception and the recommendation/metadata
    fields are left null.
    """
    total_time_ms = (time.perf_counter() - start) * 1000
    result = result or {}
    recommendation = result.get("recommendation")

    obs_logger.info(
        "recommend_request",
        extra={
            "query": query,
            # Cap the logged recommendation so a long reply can't bloat logs.
            "recommendation": recommendation[:200] if recommendation else None,
            "retrieved_ids": result.get("retrieved_ids"),
            "retrieval_time_ms": result.get("retrieval_time_ms"),
            "llm_time_ms": result.get("llm_time_ms"),
            "total_time_ms": total_time_ms,
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "total_tokens": result.get("total_tokens"),
            "status": "error" if error is not None else "success",
            "error_type": type(error).__name__ if error is not None else None,
        },
    )
