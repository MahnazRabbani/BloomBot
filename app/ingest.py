"""Ingest the bouquet catalog into ChromaDB.

Reads ``app/catalog.json``, builds a natural-language embedding string for each
bouquet, embeds it with OpenAI ``text-embedding-3-small``, and stores the vectors
(plus metadata) in a persistent ChromaDB collection named ``bouquets``.

Run from the repo root:

    python -m app.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

# Paths are resolved relative to the repo root so the script works no matter
# the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "app" / "catalog.json"
CHROMA_DIR = REPO_ROOT / "chroma_db"

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "bouquets"


def build_embedding_text(bouquet: dict) -> str:
    """Compose a semantically rich string for a single bouquet.

    We fold in the fields that carry recommendation-relevant meaning (name,
    occasions, mood, colors, flowers, symbolism, description) so that a user's
    natural-language query embeds close to the bouquets that fit it.
    """
    colors = ", ".join(bouquet.get("colors", []))
    flowers = ", ".join(bouquet.get("flowers", []))
    occasions = ", ".join(bouquet.get("occasions", []))
    mood = ", ".join(bouquet.get("mood", []))

    parts = [
        f"Name: {bouquet['name']}.",
        f"Occasions: {occasions}." if occasions else "",
        f"Mood: {mood}." if mood else "",
        f"Colors: {colors}." if colors else "",
        f"Flowers: {flowers}." if flowers else "",
        f"Symbolism: {bouquet['symbolism']}" if bouquet.get("symbolism") else "",
        f"Description: {bouquet['description']}" if bouquet.get("description") else "",
    ]
    return " ".join(part for part in parts if part)


def build_metadata(bouquet: dict) -> dict:
    """Flatten a bouquet into ChromaDB-compatible metadata.

    ChromaDB metadata values must be scalars, so list fields are joined into
    comma-separated strings. The full record is kept under ``catalog_json`` so
    the original structure can be recovered at query time.
    """
    return {
        "id": bouquet["id"],
        "name": bouquet["name"],
        "price": bouquet["price"],
        "size": bouquet.get("size", ""),
        "stem_count": bouquet.get("stem_count", 0),
        "colors": ", ".join(bouquet.get("colors", [])),
        "flowers": ", ".join(bouquet.get("flowers", [])),
        "occasions": ", ".join(bouquet.get("occasions", [])),
        "mood": ", ".join(bouquet.get("mood", [])),
        "catalog_json": json.dumps(bouquet),
    }


def load_bouquets() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    bouquets = data["bouquets"]
    if not bouquets:
        raise ValueError(f"No bouquets found in {CATALOG_PATH}")
    return bouquets


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in a single API call, preserving input order."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest() -> None:
    load_dotenv()

    bouquets = load_bouquets()
    print(f"Loaded {len(bouquets)} bouquets from {CATALOG_PATH}")

    documents = [build_embedding_text(b) for b in bouquets]
    metadatas = [build_metadata(b) for b in bouquets]
    ids = [b["id"] for b in bouquets]

    openai_client = OpenAI()
    print(f"Embedding {len(documents)} bouquets with {EMBEDDING_MODEL}...")
    embeddings = embed_texts(openai_client, documents)

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Recreate the collection so re-runs stay idempotent instead of appending
    # duplicates.
    chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(
        f"Ingested {collection.count()} bouquets into collection "
        f"'{COLLECTION_NAME}' at {CHROMA_DIR}"
    )


if __name__ == "__main__":
    ingest()

