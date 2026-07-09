# BloomBot

An AI-powered bouquet recommendation API. Describe what you need in natural language, get back a personalized flower recommendation grounded in a real product catalog.

**Live demo:** https://bloombot-ifh6.onrender.com/docs

## What it does

A customer describes an occasion, mood, or preference in plain language (e.g. *"something for my mother's 60th birthday, she loves purple, budget around $80"*), and the system retrieves the most relevant bouquets from a catalog and explains why each one fits, using retrieval-augmented generation (RAG) so recommendations are grounded in real inventory rather than invented by the LLM.

## Architecture

```
Customer query
     ↓
Embed query (OpenAI text-embedding-3-small)
     ↓
Retrieve top-k similar bouquets (ChromaDB, cosine similarity)
     ↓
Build grounded prompt (retrieved bouquets as context)
     ↓
Generate recommendation (OpenAI gpt-4o-mini)
     ↓
Return via FastAPI endpoint
```

The catalog (30 bouquets) is embedded once at ingestion time and stored in a persistent ChromaDB vector store. Each customer query is embedded with the same model, and the nearest matches are retrieved by semantic similarity, then passed to the LLM as grounding context so it can only recommend from real, existing products.

## Tech stack

- **Language:** Python
- **API framework:** FastAPI
- **LLM:** OpenAI gpt-4o-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector store:** ChromaDB
- **Testing:** pytest
- **Rate limiting:** slowapi
- **Containerization:** Docker
- **Deployment:** Render

## Running locally

```bash
git clone https://github.com/MahnazRabbani/BloomBot.git
cd BloomBot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-key-here
```

Ingest the catalog into the vector store:
```bash
python -m app.ingest
```

Start the API:
```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive API docs.

## Running with Docker

```bash
docker build -t bloombot .
docker run -p 8000:8000 --env-file .env bloombot
```

## API

**POST /recommend**

Request:
```json
{ "query": "romantic flowers for an anniversary" }
```

Response (200):
```json
{ "recommendation": "..." }
```

Validation and limits:
- `query` must be 1-500 characters and not blank/whitespace-only (422 if violated)
- Rate limited to 10 requests per minute per client IP (429 if exceeded)

Error responses:
- `422` — invalid request (empty/blank/too-long query, malformed JSON)
- `429` — rate limit exceeded
- `503` — upstream OpenAI rate limit hit, retry later
- `500` — internal server error (details are logged server-side, never exposed in the response)

**GET /**

Health check. Returns `{ "status": "ok", "service": "BloomBot" }`.

## Testing

```bash
pytest -v
```

18 tests covering:
- Unit tests for the retriever (semantic search, empty query/collection handling)
- Unit tests for the RAG chain (prompt construction, empty retrieval fallback, malformed LLM response handling)
- Integration tests for the FastAPI endpoint (validation, rate limiting, error handling, no internal detail leakage on failure)

All external dependencies (OpenAI, ChromaDB) are mocked, so the suite runs in under 2 seconds with no network calls or API cost.

## Error handling

The `/recommend` endpoint distinguishes between failure types rather than treating every error identically:
- OpenAI rate limits are surfaced as a `503` with a retry-friendly message
- OpenAI authentication failures are logged server-side and returned as a generic `500` (never exposing key or config details to the client)
- Any other failure is logged with full detail server-side and returned as a generic `500`

This ensures the API never leaks internal exception messages, stack traces, or configuration details to callers.

## Project status

Phase 1 (MVP) and Phase 2 (production quality: testing, error handling, input validation, rate limiting, code cleanup) complete. See `/docs` for phase-by-phase learning notes and design decisions.
