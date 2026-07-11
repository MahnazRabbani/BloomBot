"""Evaluate BloomBot's retrieval quality against a ground-truth test set.

Runs every query in ``evals/test_queries.json`` through the real retriever
(``app.retriever.retrieve``), compares the retrieved bouquet ids against the
manually curated ``expected_ids``, and reports precision / recall / F1 both
per-query and macro-averaged across the whole set.

This is a standalone script, not a pytest test. It makes real OpenAI embedding
calls and reads the populated ChromaDB store, so it needs ``OPENAI_API_KEY`` in
the environment (loaded from ``.env`` by the retriever).

Run from the project root:

    python -m evals.eval_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path

from app.retriever import retrieve

TOP_K = 4

# Resolve data paths relative to this file so the script works regardless of the
# current working directory it's invoked from.
_EVALS_DIR = Path(__file__).resolve().parent
TEST_QUERIES_PATH = _EVALS_DIR / "test_queries.json"
RESULTS_PATH = _EVALS_DIR / "retrieval_results.json"


def _prf(retrieved_ids: list[str], expected_ids: list[str]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for one query.

    Comparison is set-based. Precision is 0 when nothing was retrieved; recall is
    0 when nothing was expected; F1 is 0 when precision and recall are both 0.
    """
    retrieved = set(retrieved_ids)
    expected = set(expected_ids)
    hits = len(retrieved & expected)

    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return precision, recall, f1


def _evaluate_query(entry: dict) -> dict:
    """Run one test-set entry through the retriever and score it."""
    query = entry["query"]
    expected_ids = entry["expected_ids"]

    retrieval = retrieve(query, k=TOP_K)
    retrieved_ids = [b["id"] for b in retrieval["results"] if "id" in b]

    precision, recall, f1 = _prf(retrieved_ids, expected_ids)

    return {
        "id": entry["id"],
        "category": entry.get("category", ""),
        "query": query,
        "retrieved_ids": retrieved_ids,
        "expected_ids": expected_ids,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "retrieval_time_ms": retrieval["retrieval_time_ms"],
    }


def _macro_average(per_query: list[dict]) -> dict:
    """Macro-average precision / recall / F1 across all queries (unweighted)."""
    n = len(per_query)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "num_queries": 0}

    return {
        "precision": sum(r["precision"] for r in per_query) / n,
        "recall": sum(r["recall"] for r in per_query) / n,
        "f1": sum(r["f1"] for r in per_query) / n,
        "num_queries": n,
    }


def _print_table(per_query: list[dict], aggregate: dict) -> None:
    """Print a fixed-width results table followed by the aggregate scores."""
    header = (
        f"{'query_id':<9}{'category':<16}{'retrieved_ids':<24}"
        f"{'expected_ids':<24}{'prec':>7}{'rec':>7}{'f1':>7}"
    )
    print(header)
    print("-" * len(header))

    for r in per_query:
        retrieved = ",".join(r["retrieved_ids"]) or "-"
        expected = ",".join(r["expected_ids"]) or "-"
        print(
            f"{r['id']:<9}{r['category']:<16}{retrieved:<24}{expected:<24}"
            f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}"
        )

    print("-" * len(header))
    print(
        f"{'MACRO AVG':<9}{'':<16}{'':<24}{'':<24}"
        f"{aggregate['precision']:>7.3f}{aggregate['recall']:>7.3f}"
        f"{aggregate['f1']:>7.3f}"
    )
    print(f"\nQueries evaluated: {aggregate['num_queries']}  (top_k={TOP_K})")
    print(
        f"Macro precision: {aggregate['precision']:.3f}  "
        f"Macro recall: {aggregate['recall']:.3f}  "
        f"Macro F1: {aggregate['f1']:.3f}"
    )


def main() -> None:
    with TEST_QUERIES_PATH.open() as f:
        test_set = json.load(f)

    queries = test_set["queries"]
    per_query = [_evaluate_query(entry) for entry in queries]
    aggregate = _macro_average(per_query)

    _print_table(per_query, aggregate)

    output = {
        "top_k": TOP_K,
        "aggregate": aggregate,
        "per_query": per_query,
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
