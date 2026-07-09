"""Turn a natural-language request into a florist's recommendation.

This is the RAG step: retrieve candidate bouquets from ChromaDB, ground a
gpt-4o-mini prompt in exactly those candidates, and return the model's written
recommendation. The model is instructed to recommend *only* from the retrieved
set so it can never invent bouquets that are not in the catalog.
"""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from app.retriever import retrieve

CHAT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

SYSTEM_PROMPT = (
    "You are a knowledgeable, warm florist's assistant for an online flower "
    "shop. You help customers choose the perfect bouquet.\n\n"
    "Rules you must follow:\n"
    "- Recommend ONLY from the bouquets provided in the context below. Never "
    "invent, assume, or mention bouquets that are not listed.\n"
    "- For each bouquet you recommend, explain specifically why it fits the "
    "customer's request, referencing its flowers, colors, occasions, mood, or "
    "symbolism.\n"
    "- If none of the provided bouquets fit well, say so honestly and "
    "recommend the closest options.\n"
    "- Mention each recommended bouquet's name and price.\n"
    "- Keep a friendly, concise tone."
)


def _format_context(bouquets: list[dict]) -> str:
    """Render retrieved bouquets into a readable block for the prompt."""
    if not bouquets:
        return "(No bouquets were found in the catalog.)"

    blocks = []
    for i, b in enumerate(bouquets, start=1):
        flowers = ", ".join(b.get("flowers", []))
        occasions = ", ".join(b.get("occasions", []))
        blocks.append(
            f"{i}. {b['name']} — ${b['price']}\n"
            f"   Flowers: {flowers}\n"
            f"   Occasions: {occasions}\n"
            f"   Symbolism: {b.get('symbolism', '')}\n"
            f"   Description: {b.get('description', '')}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(query: str, bouquets: list[dict]) -> str:
    context = _format_context(bouquets)
    return (
        "Here are the available bouquets to choose from:\n\n"
        f"{context}\n\n"
        f'Customer request: "{query}"\n\n'
        "Recommend the best-fitting bouquet(s) from the list above and explain "
        "why each one fits."
    )


def recommend(query: str) -> str:
    """Retrieve candidate bouquets and return an LLM-written recommendation."""
    load_dotenv()

    bouquets = retrieve(query, k=4)

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(query, bouquets)},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned no content")
    return content
