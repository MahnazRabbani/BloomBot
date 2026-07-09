"""Unit tests for app.chain.recommend.

retrieve() and the OpenAI client are mocked so these tests make no DB or
network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.chain import _format_context, recommend


def _make_openai_mock(content: str = "Here is my recommendation.") -> MagicMock:
    """An OpenAI() stand-in whose chat completion returns ``content``."""
    client = MagicMock()
    message = MagicMock()
    message.content = content
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=message)]
    )
    return client


SAMPLE_BOUQUETS = [
    {
        "id": "001",
        "name": "Sunset Garden",
        "price": 65,
        "flowers": ["sunflowers", "peach roses"],
        "occasions": ["birthday", "friendship"],
        "symbolism": "Sunflowers represent adoration.",
        "description": "A radiant bouquet.",
    }
]


def test_recommend_calls_retrieve_with_query():
    mock_retrieve = MagicMock(return_value=SAMPLE_BOUQUETS)

    with patch("app.chain.load_dotenv"), patch(
        "app.chain.retrieve", mock_retrieve
    ), patch("app.chain.OpenAI", return_value=_make_openai_mock()):
        recommend("something for a birthday")

    mock_retrieve.assert_called_once_with("something for a birthday", k=4)


def test_recommend_returns_llm_response_text():
    openai_mock = _make_openai_mock(content="I recommend the Sunset Garden!")

    with patch("app.chain.load_dotenv"), patch(
        "app.chain.retrieve", return_value=SAMPLE_BOUQUETS
    ), patch("app.chain.OpenAI", return_value=openai_mock):
        result = recommend("something cheerful")

    assert result == "I recommend the Sunset Garden!"


def test_recommend_handles_empty_retrieval():
    openai_mock = _make_openai_mock(content="Sorry, nothing matches.")

    with patch("app.chain.load_dotenv"), patch(
        "app.chain.retrieve", return_value=[]
    ), patch("app.chain.OpenAI", return_value=openai_mock):
        result = recommend("obscure request")

    # Should not raise, and should still return the model's text.
    assert result == "Sorry, nothing matches."

    # The prompt actually sent must contain the empty-catalog notice.
    sent_messages = openai_mock.chat.completions.create.call_args.kwargs["messages"]
    user_content = sent_messages[-1]["content"]
    assert "No bouquets were found" in user_content


def test_format_context_empty_input_does_not_crash():
    # Direct check of the helper the task calls out.
    assert _format_context([]) == "(No bouquets were found in the catalog.)"
