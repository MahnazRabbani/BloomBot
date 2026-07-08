"""FastAPI entrypoint for BloomBot.

Exposes the bouquet recommendation chain over HTTP:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chain import recommend

app = FastAPI(
    title="BloomBot",
    description="AI bouquet recommendation API.",
    version="0.1.0",
)


class RecommendRequest(BaseModel):
    query: str = Field(..., description="The customer's natural-language request.")


class RecommendResponse(BaseModel):
    recommendation: str = Field(..., description="The florist assistant's reply.")


@app.get("/")
def health_check() -> dict:
    """Simple liveness check."""
    return {"status": "ok", "service": "BloomBot"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_bouquet(request: RecommendRequest) -> RecommendResponse:
    """Return an AI-generated bouquet recommendation for the customer's query."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be empty")

    try:
        recommendation = recommend(query)
    except Exception as exc:  # noqa: BLE001 — surface any chain failure as 500
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {exc}"
        ) from exc

    return RecommendResponse(recommendation=recommendation)
