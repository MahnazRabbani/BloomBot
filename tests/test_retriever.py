"""Unit tests for app.retriever.retrieve.

The OpenAI and ChromaDB clients are fully mocked so these tests make no network
calls and need no populated vector store.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.retriever import retrieve


def _make_openai_mock() -> MagicMock:
    """An OpenAI() stand-in whose embeddings.create returns one fake vector."""
    client = MagicMock()
    embedding_item = MagicMock()
    embedding_item.embedding = [0.1, 0.2, 0.3]
    client.embeddings.create.return_value = MagicMock(data=[embedding_item])
    return client


def _make_chroma_mock(collection: MagicMock) -> MagicMock:
    """A chromadb module stand-in whose PersistentClient yields ``collection``."""
    chroma = MagicMock()
    chroma.PersistentClient.return_value.get_collection.return_value = collection
    return chroma


def test_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        retrieve("")

    with pytest.raises(ValueError):
        retrieve("   ")


def test_valid_query_returns_parsed_bouquets():
    bouquet_a = {"id": "001", "name": "Sunset Garden", "price": 65}
    bouquet_b = {"id": "004", "name": "Crimson Devotion", "price": 85}

    collection = MagicMock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "metadatas": [
            [
                {"catalog_json": json.dumps(bouquet_a)},
                {"catalog_json": json.dumps(bouquet_b)},
            ]
        ]
    }

    with patch("app.retriever.load_dotenv"), patch(
        "app.retriever.OpenAI", return_value=_make_openai_mock()
    ), patch("app.retriever.chromadb", _make_chroma_mock(collection)):
        results = retrieve("something cheerful for a birthday", k=2)

    assert results == [bouquet_a, bouquet_b]
    # Full nested dicts are returned, not the flattened metadata strings.
    assert all(isinstance(r, dict) for r in results)
    collection.query.assert_called_once()


def test_k_zero_returns_empty_list():
    collection = MagicMock()
    collection.count.return_value = 30  # collection is populated...

    with patch("app.retriever.load_dotenv"), patch(
        "app.retriever.OpenAI", return_value=_make_openai_mock()
    ), patch("app.retriever.chromadb", _make_chroma_mock(collection)):
        results = retrieve("anything", k=0)  # ...but caller asked for 0

    assert results == []
    collection.query.assert_not_called()


def test_empty_collection_returns_empty_list():
    collection = MagicMock()
    collection.count.return_value = 0  # nothing ingested yet

    with patch("app.retriever.load_dotenv"), patch(
        "app.retriever.OpenAI", return_value=_make_openai_mock()
    ), patch("app.retriever.chromadb", _make_chroma_mock(collection)):
        results = retrieve("anything", k=4)

    assert results == []
    collection.query.assert_not_called()
