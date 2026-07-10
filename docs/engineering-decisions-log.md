# Engineering Decisions Log

Design rationale and tradeoffs for BloomBot, phases 1-3. Updated per phase.

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

## Open items carried forward
- Redis-backed rate limiting if the service moves beyond a single instance
- Re-ranking or hybrid (keyword + vector) search if retrieval precision becomes a bottleneck at larger catalog sizes
- Lint/format check (ruff/black) as an additional required CI job
