"""Summarize BloomBot structured request logs from stdin.

Reads the JSON log lines emitted by the ``bloombot`` logger (see
``app/logging_config.py``) — one JSON object per line — keeps the per-request
``recommend_request`` entries, and prints a summary: request/error counts,
latency distributions, token usage, an estimated OpenAI cost, and the time span
covered.

It is defensive by design: non-JSON lines, lines with the wrong ``message``,
and records missing individual fields are all tolerated. Each statistic is
computed only over the records that actually carry the relevant field.

This is a standalone utility, not a pytest test.

Usage:

    cat logs.json | python scripts/analyze_logs.py

    # or in production, pipe a live/rotated log through it:
    render logs --tail 10000 | python scripts/analyze_logs.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime

TARGET_MESSAGE = "recommend_request"

# Latency fields summarized with the full distribution (mean/median/p95/max).
LATENCY_FIELDS = ["total_time_ms", "retrieval_time_ms", "llm_time_ms"]

# Token fields summarized as per-request means (and totaled for cost).
TOKEN_FIELDS = ["prompt_tokens", "completion_tokens", "total_tokens"]

# OpenAI pricing, USD per 1M tokens.
PRICE_GPT4O_MINI_INPUT = 0.15
PRICE_GPT4O_MINI_OUTPUT = 0.60
PRICE_EMBEDDING_SMALL = 0.02

# Rough chars-per-token ratio for estimating embedding tokens from the query
# text, since embedding token counts are not recorded in the logs.
_CHARS_PER_TOKEN = 4

# Timestamp formats produced by python-json-logger's default asctime, plus a
# couple of common fallbacks, tried in order.
_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
]


def _as_number(value) -> float | None:
    """Return ``value`` as a float if it is a real number, else ``None``.

    Booleans are rejected (``True``/``False`` are ints in Python but never a
    valid latency or token count).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_timestamp(value) -> datetime | None:
    """Parse a log timestamp string into a datetime, or ``None`` if unrecognized."""
    if not isinstance(value, str):
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of a pre-sorted, non-empty list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def _distribution(values: list[float]) -> dict | None:
    """Return mean/median/p95/max/count for a list of numbers, or ``None`` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
        "count": len(ordered),
    }


def _read_records(stream) -> tuple[list[dict], int, int]:
    """Parse JSON-per-line input into recommend_request records.

    Returns ``(records, total_lines, malformed_lines)`` where ``records`` are the
    dict entries whose ``message`` matches ``TARGET_MESSAGE``.
    """
    records: list[dict] = []
    total_lines = 0
    malformed = 0

    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        total_lines += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(obj, dict) and obj.get("message") == TARGET_MESSAGE:
            records.append(obj)

    return records, total_lines, malformed


def _collect_values(records: list[dict], field: str) -> list[float]:
    """Numeric values of ``field`` across records that carry it as a number."""
    values = []
    for rec in records:
        num = _as_number(rec.get(field))
        if num is not None:
            values.append(num)
    return values


def _estimate_cost(records: list[dict]) -> dict:
    """Estimate total OpenAI spend across the analyzed requests.

    LLM cost comes from the logged gpt-4o-mini prompt/completion token counts.
    Embedding cost is *approximated* from the query text, since embedding token
    counts are not logged.
    """
    prompt_tokens = sum(_collect_values(records, "prompt_tokens"))
    completion_tokens = sum(_collect_values(records, "completion_tokens"))

    embedding_tokens = 0
    for rec in records:
        query = rec.get("query")
        if isinstance(query, str) and query:
            embedding_tokens += math.ceil(len(query) / _CHARS_PER_TOKEN)

    input_cost = prompt_tokens / 1_000_000 * PRICE_GPT4O_MINI_INPUT
    output_cost = completion_tokens / 1_000_000 * PRICE_GPT4O_MINI_OUTPUT
    embedding_cost = embedding_tokens / 1_000_000 * PRICE_EMBEDDING_SMALL

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "embedding_tokens_est": embedding_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "embedding_cost": embedding_cost,
        "total_cost": input_cost + output_cost + embedding_cost,
    }


def _time_range(records: list[dict]) -> tuple[str, str] | None:
    """Earliest and latest timestamp (as original strings), or ``None``."""
    stamped = []
    for rec in records:
        raw = rec.get("timestamp")
        parsed = _parse_timestamp(raw)
        if parsed is not None:
            stamped.append((parsed, raw))
    if not stamped:
        return None
    stamped.sort(key=lambda pair: pair[0])
    return stamped[0][1], stamped[-1][1]


def _print_report(records: list[dict], malformed: int) -> None:
    total = len(records)

    # Request counts. Anything not explicitly "success" counts as an error
    # (covers status == "error" and any records with a missing/unknown status).
    successes = sum(1 for r in records if r.get("status") == "success")
    errors = total - successes
    error_rate = (errors / total * 100) if total else 0.0

    print("=" * 60)
    print("BloomBot request log analysis")
    print("=" * 60)

    print("\nRequests")
    print("-" * 60)
    print(f"  Total requests : {total}")
    print(f"  Successes      : {successes}")
    print(f"  Errors         : {errors}")
    print(f"  Error rate     : {error_rate:.1f}%")
    if malformed:
        print(f"  (skipped {malformed} non-JSON input line(s))")

    print("\nLatency (ms)")
    print("-" * 60)
    print(f"  {'metric':<20}{'mean':>10}{'median':>10}{'p95':>10}{'max':>10}")
    for field in LATENCY_FIELDS:
        stats = _distribution(_collect_values(records, field))
        if stats is None:
            print(f"  {field:<20}{'(no data)':>40}")
            continue
        print(
            f"  {field:<20}{stats['mean']:>10.1f}{stats['median']:>10.1f}"
            f"{stats['p95']:>10.1f}{stats['max']:>10.1f}"
        )

    print("\nToken usage (mean per request)")
    print("-" * 60)
    for field in TOKEN_FIELDS:
        values = _collect_values(records, field)
        if not values:
            print(f"  {field:<20}(no data)")
            continue
        mean = sum(values) / len(values)
        print(f"  {field:<20}{mean:>10.1f}   (n={len(values)})")

    cost = _estimate_cost(records)
    print("\nEstimated cost (USD)")
    print("-" * 60)
    print(
        f"  gpt-4o-mini input   : ${cost['input_cost']:.6f}  "
        f"({int(cost['prompt_tokens'])} tokens @ ${PRICE_GPT4O_MINI_INPUT}/1M)"
    )
    print(
        f"  gpt-4o-mini output  : ${cost['output_cost']:.6f}  "
        f"({int(cost['completion_tokens'])} tokens @ ${PRICE_GPT4O_MINI_OUTPUT}/1M)"
    )
    print(
        f"  embeddings (est.)   : ${cost['embedding_cost']:.6f}  "
        f"(~{int(cost['embedding_tokens_est'])} tokens @ ${PRICE_EMBEDDING_SMALL}/1M)"
    )
    print(f"  {'total':<20}: ${cost['total_cost']:.6f}")
    if total:
        print(f"  per request (avg)   : ${cost['total_cost'] / total:.6f}")
    print("  note: embedding tokens are estimated from query length (not logged).")

    print("\nTime range")
    print("-" * 60)
    span = _time_range(records)
    if span is None:
        print("  (no parseable timestamps)")
    else:
        earliest, latest = span
        print(f"  Earliest : {earliest}")
        print(f"  Latest   : {latest}")
    print()


def main() -> int:
    records, total_lines, malformed = _read_records(sys.stdin)

    if total_lines == 0:
        print("No input received on stdin.", file=sys.stderr)
        return 1

    if not records:
        print(
            f"No '{TARGET_MESSAGE}' log lines found "
            f"(read {total_lines} line(s), {malformed} malformed).",
            file=sys.stderr,
        )
        return 1

    _print_report(records, malformed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
