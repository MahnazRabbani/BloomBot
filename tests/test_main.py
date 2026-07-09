"""Integration tests for the FastAPI app in app.main.

recommend() is mocked so the endpoint is exercised end-to-end (routing,
validation, serialization) without any real DB or OpenAI calls.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_recommend_valid_query_returns_200():
    with patch("app.main.recommend", return_value="I recommend the Sunset Garden!"):
        response = client.post("/recommend", json={"query": "cheerful birthday flowers"})

    assert response.status_code == 200
    assert response.json() == {"recommendation": "I recommend the Sunset Garden!"}


def test_recommend_empty_query_returns_422():
    # recommend() should never be reached for an empty query.
    with patch("app.main.recommend") as mock_recommend:
        response = client.post("/recommend", json={"query": "   "})

    assert response.status_code == 422
    mock_recommend.assert_not_called()


def test_health_check_returns_200():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "BloomBot"}


def test_recommend_chain_failure_returns_500():
    with patch("app.main.recommend", side_effect=RuntimeError("OpenAI is down")):
        response = client.post("/recommend", json={"query": "anything"})

    assert response.status_code == 500
    assert "detail" in response.json()
    assert "OpenAI is down" in response.json()["detail"]
