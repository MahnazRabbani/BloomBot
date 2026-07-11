"""Retrieve bouquets from ChromaDB by semantic similarity.

Embeds a natural-language query with the same OpenAI model used at ingest time
(``text-embedding-3-small``) and returns the top-k closest bouquets from the
persistent ``bouquets`` collection, parsed back into their original catalog
structure.
"""

from __future__ import annotations

import json
import time

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from app.ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def _embed_query(client: OpenAI, query: str) -> list[float]:
    """Embed a single query string, matching the ingest-time model."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def retrieve(query: str, k: int = 4) -> dict:
    """Retrieve the top-``k`` bouquets most similar to ``query``.

    Returns a dict with:

    - ``results``: the matching bouquets, each the original record parsed from
      the ``catalog_json`` metadata field so callers get the full nested
      structure (components, symbolism, etc.) rather than the flattened
      metadata.
    - ``retrieval_time_ms``: wall-clock time spent embedding the query and
      querying ChromaDB, in milliseconds.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")

    start = time.perf_counter()

    load_dotenv()

    openai_client = OpenAI()
    query_embedding = _embed_query(openai_client, query)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    # Never ask Chroma for more results than exist, or it raises.
    n_results = min(k, collection.count())
    if n_results == 0:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"results": [], "retrieval_time_ms": elapsed_ms}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    # query() returns lists nested one level per query embedding; we sent one.
    metadatas = results["metadatas"][0]
    bouquets = [json.loads(meta["catalog_json"]) for meta in metadatas]

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"results": bouquets, "retrieval_time_ms": elapsed_ms}
