# BloomBot

An AI-powered bouquet recommendation API. Describe what you need in natural language, get back a personalized flower recommendation grounded in a real product catalog.

**Live demo:** [https://bloombot-ifh6.onrender.com/docs]/docs

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

Response:
```json
{ "recommendation": "..." }
```

## Project status

Phase 1 (MVP) complete: working RAG pipeline, deployed API. See `/docs` for phase-by-phase learning notes and design decisions.
