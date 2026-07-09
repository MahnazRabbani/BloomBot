"""FastAPI entrypoint for BloomBot.

Exposes the bouquet recommendation chain over HTTP:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

import openai
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.chain import recommend

logger = logging.getLogger(__name__)

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


class RecommendResponse(BaseModel):
    recommendation: str = Field(..., description="The florist assistant's reply.")


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
    try:
        recommendation = recommend(payload.query)
    except openai.RateLimitError as exc:
        # Transient upstream capacity issue — ask the caller to retry later.
        # No internal details are exposed in the response.
        logger.warning("OpenAI rate limit hit: %s", exc)
        raise HTTPException(
            status_code=503, detail="Service temporarily busy, please try again later."
        ) from exc
    except openai.AuthenticationError as exc:
        # Misconfigured/invalid API key — a server-side problem. Log the real
        # cause for operators, but never leak it to the client.
        logger.error("OpenAI authentication failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from exc
    except Exception as exc:  # noqa: BLE001 — generic fallback
        logger.exception("Recommendation failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Internal server error."
        ) from exc

    return RecommendResponse(recommendation=recommendation)
