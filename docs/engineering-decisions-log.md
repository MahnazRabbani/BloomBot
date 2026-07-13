# Engineering Decisions Log

Design rationale and tradeoffs for BloomBot. Updated per phase.

---

## Phase 1 — RAG Pipeline, API, Deployment

**Scope:** Catalog ingestion into a vector store, semantic retrieval, LLM-generated recommendations grounded in retrieved results, FastAPI endpoint, Docker packaging, live deployment.

### Decisions

| Decision | Rationale |
|---|---|
| `text-embedding-3-small` | Lowest-cost OpenAI embedding model; sufficient quality for a 30-item catalog |
| `gpt-4o-mini`, temperature 0.2 | Lowest-cost capable chat model; low temperature favors consistent, grounded output over variation |
| ChromaDB | Embedded, no separate infrastructure; appropriate at this catalog size |
| Render | Free tier, native Docker support, auto-deploy from GitHub without a separate CI/CD setup |
| Bouquet schema includes a `components` field (flowers + quantities), unused in Phase 1 | Anticipates Phase 6 (inventory-aware composition); avoids a schema migration later |
| Vector store baked into the Docker image rather than ingested on container start | Simpler and cheaper for a static demo catalog; tradeoff is the image must be rebuilt after catalog changes |

### System design

Retrieval-augmented generation: the query is embedded with the same model used at ingestion time, so both live in a comparable vector space. Top-k nearest catalog entries (cosine similarity) are retrieved and injected into the LLM prompt as the only permitted source of truth — the system prompt explicitly constrains the model to recommend only from the retrieved set, which is the primary anti-hallucination mechanism in this architecture.

Retrieval quality is a direct function of what text is embedded at ingestion time; catalog entries were written with occasion, mood, and symbolism language rather than bare attributes, matching the vocabulary a customer query is likely to use.

### Notes
`.env` was briefly committed to git (caught by GitHub push protection before reaching a shared state); resolved via `git rm --cached` and a commit amend. Underlines that secrets exposure risk exists at the local-commit stage, not just at push time.

---

## Phase 2 — Testing, Error Handling, Validation, Rate Limiting

**Scope:** Unit and integration test coverage, differentiated exception handling, request validation, per-IP rate limiting, dead-code removal.

### Decisions

| Decision | Rationale |
|---|---|
| pytest + `unittest.mock`, all external calls (OpenAI, ChromaDB) mocked | Full suite (18 tests) runs in ~1.5s with no network dependency or API cost |
| Differentiated exception handling: `RateLimitError` → 503, `AuthenticationError` → 500 (logged), generic `Exception` → 500 (logged) | Transient failures (rate limits) and configuration failures (auth) warrant different client-facing responses; both are logged server-side with full detail, never exposed in the response body |
| Pydantic `Field(min_length=1, max_length=500)` plus a `field_validator` | Length constraints alone don't reject whitespace-only input (`"   "` satisfies `min_length=1`); the validator closes that gap |
| slowapi, 10 req/min per IP, in-memory backend | Purpose-built for FastAPI; no external dependency justified at single-instance scale |
| Removed unused `app/api/`, `app/core/`, `app/models/`, `app/services/` scaffolding | Flat `app/` structure matches actual size (4 modules); premature subpackaging deferred until Phase 6 justifies it |

### System design

Error handling follows a specific-before-general pattern: known upstream failure modes are caught first and mapped to distinct status codes, with an unqualified `Exception` catch as the terminal fallback. No exception message, stack trace, or internal detail is ever included in an HTTP response — verified directly in tests via negative assertions (e.g. asserting the raw error string is absent from the response body, not just that a generic message is present).

Test isolation required an autouse fixture to disable the rate limiter's shared in-memory state for unrelated tests, since a stateful global counter can otherwise cause order-dependent test failures.

### Notes
Removing the "redundant-looking" manual blank-query check during the Pydantic migration surfaced a real regression (whitespace passing `min_length`) — caught because the test suite was in place before the refactor, not after.

---

## Phase 3 — CI/CD, Branch Protection

**Scope:** Automated test execution on push, protected `main` branch, verified auto-deploy, CI status badge.

### Decisions

| Decision | Rationale |
|---|---|
| GitHub Actions | No separate CI account/service; integrates natively with branch protection status checks |
| Trigger on `push` and `pull_request`, any branch | Surfaces failures at the earliest point, not only on `main` |
| `cache: pip` in `setup-python` | Near-zero-cost build speedup keyed on `requirements.txt` hash |
| Branch protection: PR required, `test` status check required before merge | Enforces verification before code reaches `main`; the same discipline applies regardless of team size |
| Render "On Commit" auto-deploy, verified against deploy history rather than assumed from the settings toggle | Configuration state and actual behavior can diverge; checked evidence, not just the setting |

### System design

CI runs the full suite on a clean VM per run, which validates the same class of problem Docker solves for runtime — environment-dependent success — but for verification rather than execution. Branch protection was only enabled after the CI check existed, since a required check has no meaning without something to require. The full loop (branch → PR → passing check → merge) was exercised at least once post-setup to confirm protection actually gates merges rather than only appearing configured.

---
## Phase 4 — Monitoring + Evaluation

**Scope:** Structured per-request logging, retrieval quality evaluation (precision/recall/F1), end-to-end LLM-as-judge evaluation, log analysis utility.

### Decisions

| Decision | Rationale |
|---|---|
| `python-json-logger` for structured logging, not a managed observability platform (Datadog, LangSmith) | Demonstrates understanding of what to measure and why; a managed platform adds cost and setup without adding portfolio signal. Other options: LangSmith (LangChain-coupled, not in our stack), Datadog/Grafana (production-grade but overkill for a demo deployment), OpenTelemetry (standard but heavier setup) |
| Separate `bloombot` JSON logger alongside the existing stdlib logger | Operator-facing error messages (human-readable) and machine-parseable observability data serve different consumers; merging them forces a format compromise. In a larger system, everything would be JSON for uniform aggregation |
| `time.perf_counter()` for latency measurement | Monotonic clock with the highest available resolution; immune to system clock adjustments, unlike `time.time()` |
| Return metadata dict from `recommend()` and `retrieve()` instead of logging inside those functions | Keeps logging concerns in the API layer (`main.py`), not the domain logic; the functions remain testable without log-capture fixtures |
| 25-query eval test set with manual ground-truth IDs, not auto-generated | Ground truth must reflect human judgment about what bouquets genuinely fit a query; automated assignment would just replicate the retriever's own biases |
| Retrieval eval and e2e eval as separate standalone scripts, not pytest tests | They make real API calls, cost money, and take minutes to run; mixing them with the fast, free, mocked unit suite would discourage running tests frequently |
| `gpt-4o` as the LLM judge, not `gpt-4o-mini` | The judge should be at least as capable as the model being evaluated (`gpt-4o-mini`); using the same model to judge itself would conflate generation quality with evaluation quality. Other options: Claude (cross-provider judging, adds a second API key), human evaluation (gold standard but not repeatable or automatable) |
| Judge temperature 0 | Maximizes scoring consistency across runs; creative variation in a rubric evaluator introduces noise |

### System design

Observability is structured as a single JSON log line per request, emitted in `main.py` after every exit path (success and all error branches). The `_log_request()` helper guarantees no request is silently dropped from metrics. Fields capture timing at two granularities (retrieval vs. LLM, plus total), token usage from the OpenAI response object, and the IDs of retrieved bouquets for post-hoc retrieval debugging. The client-facing API response shape is unchanged; all observability data is internal.

Evaluation separates retrieval quality from generation quality because they fail independently. Retrieval eval uses set-based precision/recall/F1 against ground-truth IDs. E2e eval uses an LLM judge scoring five criteria on a 1-5 ordinal scale, macro-averaged per criterion.

### Key findings

Retrieval baseline: macro recall 0.81, precision 0.41, F1 0.52. Recall is strong (the right bouquet appears in the top 4 for most queries). Precision is structurally bounded (queries with 1 expected ID can never exceed 0.25 precision at k=4). The complete failure case is q13 ("birthday flowers, budget around $80"): retriever returned zero correct bouquets because semantic search has no mechanism for numerical constraints like price.

E2e baseline: 4.98/5.00 overall. Near-perfect scores across all five criteria. The critical insight: q13 scored 5/5 on the e2e eval despite 0/0/0 on retrieval. The LLM wrote a fluent, well-structured recommendation about the wrong bouquets, and the judge rated it highly. This demonstrates that fluent generation masks retrieval failures, and evaluation pipelines must assess both stages independently.

Category-level pattern: aesthetic queries perform best on retrieval (embeddings excel at mood/color language). Constraint queries perform worst (price, size, exclusions are structured filters, not semantic concepts). This directly motivates hybrid retrieval (metadata filtering + vector search) as a future improvement.

### Notes
The `retrieve()` and `recommend()` return types changed from simple values to dicts carrying metadata. This is a mild API contract break, caught and updated across all callers and tests in the same commit. In a multi-team codebase, this would warrant a deprecation path; in a single-developer project, a single atomic commit is sufficient.

## Phase 5 — UI

**Scope:** A demo-able chat interface for BloomBot, deployed independently and calling the existing API over HTTP.

### Decisions

| Decision | Rationale |
|---|---|
| Streamlit for the UI | Python-native (no separate frontend toolchain), built-in chat components (`st.chat_message`/`st.chat_input`), free hosting on Streamlit Community Cloud. Other options: Gradio (similar, but chat ergonomics and hosting story weaker for this case), Chainlit (chat-first but heavier and less general), React (most flexible and most production-realistic, but a full frontend build/deploy pipeline is disproportionate for a portfolio demo) |
| Two-service architecture: UI calls the API over HTTP rather than importing `chain.recommend()` directly | Keeps the frontend/backend boundary visible and demonstrates real API consumption — the same way an external client would integrate. The two services deploy, scale, and version independently. Importing the chain would have been simpler but would collapse the separation the project is meant to showcase |
| `BLOOMBOT_API_URL` env var for the API target, defaulting to `http://localhost:8000` | One codebase points at the local API in dev and the Render API in production with no code change; localhost stays as a dev-only default. Avoids hardcoding a deployment URL into source |
| `RecommendResponse` contract change: added a nested `meta` object (`retrieval_time_ms`, `llm_time_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`) | **Deliberate reversal, not drift.** Phase 4 explicitly kept observability metadata out of the client response (logs only). Phase 5 reverses that: surfacing latency and token/cost signals in the UI is a portfolio asset — it shows observability awareness to AI-engineering reviewers. `retrieved_ids` stays logs-only, so the reversal is scoped to what a viewer benefits from seeing, not a wholesale exposure of internals |
| Separate `ui/requirements.txt` (only `streamlit` + `requests`, pinned to the root versions) | Streamlit Cloud installs from the requirements file next to the app. A UI-only file keeps that deploy lean — the API's heavy deps (chromadb, openai, langchain, etc.) are irrelevant to a thin HTTP client and would bloat the build |
| Env var injection via Streamlit Cloud secrets (TOML in the dashboard) | Streamlit Community Cloud exposes dashboard secrets as environment variables, so the app reads `BLOOMBOT_API_URL` through `os.environ` without Streamlit-specific code (`st.secrets`). Keeps the UI code deployment-agnostic — it runs the same locally with a shell env var |

### System design

The UI is a thin client: it owns presentation and conversation-history state (`st.session_state`) but no domain logic. All retrieval and generation stays server-side in the API. This is the standard separation a real product would use, and it means the API can be consumed by other clients (a future mobile app, a partner integration) without change.

Streamlit's execution model re-runs the whole script on every interaction, so message history must live in `st.session_state` and be replayed each run; per-message metadata is stored alongside the text so the **Details** expander survives re-runs.

### Notes

`chroma_db/chroma.sqlite3` had been gitignored while the HNSW vector-index binaries were tracked. The Render deploy therefore shipped vector segments but not the SQLite file that registers the `bouquets` collection, so ChromaDB failed to find the collection at runtime. Fix was to un-ignore and commit the file. Root cause: a persistent Chroma store is a *directory* (metadata DB + per-collection index files), and gitignoring one part silently produces an incomplete store that only fails on a fresh deploy, not locally where the file already exists. Committing the store works for a small static catalog; a production path would rebuild it at container start via `app.ingest` instead.

---

## Open items carried forward
- Redis-backed rate limiting if the service moves beyond a single instance
- Re-ranking or hybrid (keyword + vector) search if retrieval precision becomes a bottleneck at larger catalog sizes
- Lint/format check (ruff/black) as an additional required CI job
