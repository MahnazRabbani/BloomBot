"""End-to-end evaluation of BloomBot recommendations using an LLM judge.

For every query in ``evals/test_queries.json`` this script:

1. Calls ``app.chain.recommend`` to produce the full recommendation.
2. Separately calls ``app.retriever.retrieve`` to reconstruct the bouquet
   context that was fed to the LLM, so the judge can check grounding.
3. Sends the query, the recommendation, and the retrieved context to ``gpt-4o``
   acting as a rubric-driven judge, and parses its 1-5 scores.

Scores are reported per-query and macro-averaged per criterion, plus an overall
average across all criteria.

This is a standalone script, not a pytest test. It makes real OpenAI calls
(recommendation model + judge) and reads the populated ChromaDB store, so it
needs ``OPENAI_API_KEY`` in the environment (loaded from ``.env``).

Run from the project root:

    python -m evals.eval_e2e
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.chain import _format_context, recommend
from app.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_e2e")

TOP_K = 4
JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0

# The rubric criteria, in the order they're scored, printed, and averaged.
CRITERIA = ["relevance", "grounding", "explanation_quality", "completeness", "tone"]

# Short column headers for the per-query table, keyed by criterion.
_COLUMN_LABELS = {
    "relevance": "rel",
    "grounding": "gnd",
    "explanation_quality": "expl",
    "completeness": "comp",
    "tone": "tone",
}

# Resolve data paths relative to this file so the script works regardless of the
# current working directory it's invoked from.
_EVALS_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = _EVALS_DIR / "test_queries.json"
RESULTS_PATH = _EVALS_DIR / "e2e_results.json"

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator of a flower-shop assistant's bouquet "
    "recommendations. You score each recommendation against a fixed rubric and "
    "you never invent facts about the bouquets beyond what you are given."
)

JUDGE_RUBRIC = (
    "Score the recommendation on each criterion using an integer from 1 (worst) "
    "to 5 (best):\n"
    "- relevance: Does the recommendation address the customer's request?\n"
    "- grounding: Does it recommend ONLY bouquets present in the retrieved "
    "context, without inventing any bouquet not listed?\n"
    "- explanation_quality: Does it explain why each recommended bouquet fits "
    "the request?\n"
    "- completeness: Does it mention each recommended bouquet's name and price?\n"
    "- tone: Is it warm, professional, and appropriate for the occasion?"
)


def _build_judge_prompt(query: str, recommendation: str, context: str) -> str:
    """Assemble the user-message content sent to the judge model."""
    return (
        f"{JUDGE_RUBRIC}\n\n"
        "=== CUSTOMER REQUEST ===\n"
        f"{query}\n\n"
        "=== RETRIEVED BOUQUET CONTEXT (the only bouquets the assistant was "
        "allowed to recommend) ===\n"
        f"{context}\n\n"
        "=== ASSISTANT RECOMMENDATION ===\n"
        f"{recommendation}\n\n"
        "Respond with ONLY a JSON object and nothing else, in exactly this "
        "shape:\n"
        '{"relevance": int, "grounding": int, "explanation_quality": int, '
        '"completeness": int, "tone": int}'
    )


def _parse_judge_scores(raw: str) -> dict | None:
    """Parse the judge's reply into a scores dict, or ``None`` if invalid.

    Tolerates the model wrapping its JSON in prose or markdown code fences by
    extracting the outermost ``{...}`` span before parsing. Returns ``None`` if
    the payload can't be parsed or is missing any rubric criterion.
    """
    if raw is None:
        return None

    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    scores = {}
    for criterion in CRITERIA:
        value = parsed.get(criterion)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        scores[criterion] = value
    return scores


def _judge(
    client: OpenAI, query: str, recommendation: str, context: str
) -> dict | None:
    """Ask the judge model to score one recommendation. ``None`` on parse failure."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_judge_prompt(query, recommendation, context)},
        ],
    )
    return _parse_judge_scores(response.choices[0].message.content)


def _evaluate_query(client: OpenAI, entry: dict) -> dict | None:
    """Run one test-set entry end-to-end and score it. ``None`` if skipped."""
    query = entry["query"]

    result = recommend(query)
    recommendation = result["recommendation"]

    # Reconstruct the context the LLM was grounded on, rendered the same way
    # app.chain builds it, so the judge sees exactly those bouquets.
    retrieval = retrieve(query, k=TOP_K)
    context = _format_context(retrieval["results"])

    scores = _judge(client, query, recommendation, context)
    if scores is None:
        logger.warning("Judge response unparseable for %s; skipping query.", entry["id"])
        return None

    avg = sum(scores[c] for c in CRITERIA) / len(CRITERIA)
    return {
        "id": entry["id"],
        "category": entry.get("category", ""),
        "query": query,
        "recommendation": recommendation,
        "retrieved_ids": result["retrieved_ids"],
        "scores": scores,
        "avg": avg,
    }


def _macro_average(per_query: list[dict]) -> dict:
    """Macro-average each criterion (and the overall mean) across scored queries."""
    n = len(per_query)
    if n == 0:
        return {
            "per_criterion": {c: 0.0 for c in CRITERIA},
            "overall": 0.0,
            "num_queries": 0,
        }

    per_criterion = {
        c: sum(r["scores"][c] for r in per_query) / n for c in CRITERIA
    }
    overall = sum(per_criterion.values()) / len(CRITERIA)
    return {
        "per_criterion": per_criterion,
        "overall": overall,
        "num_queries": n,
    }


def _print_table(per_query: list[dict], aggregate: dict) -> None:
    """Print a fixed-width per-query score table and the aggregate scores."""
    labels = [_COLUMN_LABELS[c] for c in CRITERIA]
    header = f"{'query_id':<9}{'category':<16}" + "".join(f"{lab:>6}" for lab in labels)
    header += f"{'avg':>7}"
    print(header)
    print("-" * len(header))

    for r in per_query:
        row = f"{r['id']:<9}{r['category']:<16}"
        row += "".join(f"{r['scores'][c]:>6d}" for c in CRITERIA)
        row += f"{r['avg']:>7.2f}"
        print(row)

    print("-" * len(header))
    macro = f"{'MACRO AVG':<9}{'':<16}"
    macro += "".join(f"{aggregate['per_criterion'][c]:>6.2f}" for c in CRITERIA)
    macro += f"{aggregate['overall']:>7.2f}"
    print(macro)

    print(f"\nQueries scored: {aggregate['num_queries']}  (judge={JUDGE_MODEL}, top_k={TOP_K})")
    for c in CRITERIA:
        print(f"  {c:<20} {aggregate['per_criterion'][c]:.3f}")
    print(f"Overall average across all criteria: {aggregate['overall']:.3f}")


def main() -> None:
    load_dotenv()
    client = OpenAI()

    with TEST_QUERIES_PATH.open() as f:
        test_set = json.load(f)

    queries = test_set["queries"]
    per_query = []
    for entry in queries:
        scored = _evaluate_query(client, entry)
        if scored is not None:
            per_query.append(scored)

    aggregate = _macro_average(per_query)
    _print_table(per_query, aggregate)

    skipped = len(queries) - len(per_query)
    output = {
        "judge_model": JUDGE_MODEL,
        "top_k": TOP_K,
        "num_queries": len(queries),
        "num_scored": len(per_query),
        "num_skipped": skipped,
        "aggregate": aggregate,
        "per_query": per_query,
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    if skipped:
        print(f"\nSkipped {skipped} query/queries due to unparseable judge output.")
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
