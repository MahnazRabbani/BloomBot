"""Integration tests for the FastAPI app in app.main.

recommend() is mocked so the endpoint is exercised end-to-end (routing,
validation, serialization) without any real DB or OpenAI calls.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, limiter

client = TestClient(app)


@pytest.fixture(autouse=True)
def _disable_rate_limit():
    """Disable rate limiting for every test by default.

    slowapi's counter is in-memory and shared across the whole app instance,
    so without this, requests from unrelated tests would accumulate against the
    same per-IP limit and cause flaky 429s. The dedicated 429 test re-enables
    the limiter itself.
    """
    limiter.enabled = False
    yield
    limiter.enabled = True


def _make_openai_error(error_cls, message: str):
    """Construct an openai APIStatusError subclass with a minimal fake response.

    RateLimitError/AuthenticationError require a ``response`` and ``body``, so we
    hand them a bare httpx.Response rather than hitting the network.
    """
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    status_code = 429 if error_cls is openai.RateLimitError else 401
    response = httpx.Response(status_code=status_code, request=request)
    return error_cls(message, response=response, body=None)


def test_recommend_valid_query_returns_200():
    with patch("app.main.recommend", return_value="I recommend the Sunset Garden!"):
        response = client.post("/recommend", json={"query": "cheerful birthday flowers"})

    assert response.status_code == 200
    assert response.json() == {"recommendation": "I recommend the Sunset Garden!"}


def test_recommend_empty_query_returns_422():
    # An empty query fails Pydantic min_length validation before the handler
    # runs, so recommend() is never reached.
    with patch("app.main.recommend") as mock_recommend:
        response = client.post("/recommend", json={"query": ""})

    assert response.status_code == 422
    mock_recommend.assert_not_called()


def test_recommend_whitespace_only_query_returns_422():
    # A whitespace-only query passes min_length but is rejected by the
    # field validator, so recommend() is never reached.
    with patch("app.main.recommend") as mock_recommend:
        response = client.post("/recommend", json={"query": "   "})

    assert response.status_code == 422
    mock_recommend.assert_not_called()


def test_recommend_query_too_long_returns_422():
    # A query over the 500-char max fails Pydantic validation.
    with patch("app.main.recommend") as mock_recommend:
        response = client.post("/recommend", json={"query": "a" * 501})

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
    detail = response.json()["detail"]
    assert detail == "Internal server error."
    # Internal exception details must never leak to the client.
    assert "OpenAI is down" not in detail


def test_recommend_rate_limit_returns_503():
    error = _make_openai_error(openai.RateLimitError, "rate limit exceeded")
    with patch("app.main.recommend", side_effect=error):
        response = client.post("/recommend", json={"query": "anything"})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "Service temporarily busy, please try again later."
    # No internal details leaked.
    assert "rate limit exceeded" not in detail


def test_recommend_auth_error_returns_generic_500():
    error = _make_openai_error(openai.AuthenticationError, "invalid api key")
    with patch("app.main.recommend", side_effect=error):
        response = client.post("/recommend", json={"query": "anything"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "Internal server error."
    # The real auth failure must not be exposed to the client.
    assert "invalid api key" not in detail


def test_recommend_exceeds_rate_limit_returns_429():
    # Re-enable the limiter (the autouse fixture disabled it) and clear any
    # prior counts so this test starts from a clean per-IP window.
    limiter.enabled = True
    limiter.reset()

    with patch("app.main.recommend", return_value="ok"):
        # The first 10 requests in the window are allowed.
        for i in range(10):
            response = client.post("/recommend", json={"query": "flowers"})
            assert response.status_code == 200, f"request {i + 1} unexpectedly blocked"

        # The 11th exceeds the 10/minute limit.
        response = client.post("/recommend", json={"query": "flowers"})

    assert response.status_code == 429
