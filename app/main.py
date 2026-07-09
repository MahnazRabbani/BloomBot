"""FastAPI entrypoint for BloomBot.

Exposes the bouquet recommendation chain over HTTP:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging

import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.chain import recommend

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BloomBot",
    description="AI bouquet recommendation API.",
    version="0.1.0",
)


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
def recommend_bouquet(request: RecommendRequest) -> RecommendResponse:
    """Return an AI-generated bouquet recommendation for the customer's query."""
    # Length limits (1-500 chars) are enforced by RecommendRequest validation.
    try:
        recommendation = recommend(request.query)
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
