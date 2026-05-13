# SupportOps AI Agent

Production-like pet project for an AI support agent built with FastAPI, PostgreSQL,
Qdrant, Redis, Ollama, and local embeddings.

## Current stage

Current implementation includes:

- FastAPI application factory
- `GET /` root endpoint
- `GET /health` health endpoint
- settings loaded from `.env` with `pydantic-settings`
- async SQLAlchemy session setup
- SQLAlchemy models for documents, chunks, conversations, messages, agent runs, and tool calls
- Alembic migration setup
- Docker Compose services for PostgreSQL, Qdrant, and Redis
- basic pytest coverage for root and health endpoints

Business logic is intentionally not implemented yet.

## Requirements

- Python 3.12
- Docker and Docker Compose
- Ollama installed locally for future LLM integration

## Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure environment

```bash
cp .env.example .env
```

## Start infrastructure

```bash
docker compose up -d
```

This starts:

- PostgreSQL on `localhost:5432`
- Qdrant on `localhost:6333`
- Redis on `localhost:6379`

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

## Run FastAPI

```bash
uvicorn app.main:app --reload
```

Open:

- API root: http://localhost:8000/
- Health check: http://localhost:8000/health
- OpenAPI docs: http://localhost:8000/docs

Check health from the terminal:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Apply database migrations

Start PostgreSQL first:

```bash
docker compose up -d postgres
```

Apply Alembic migrations:

```bash
alembic upgrade head
```

Check current migration:

```bash
alembic current
```

## Run tests

```bash
pytest
```

Run Qdrant integration tests after starting Qdrant:

```bash
docker compose up -d qdrant
pytest -m integration
```

## Planned next stages

1. Implement document ingestion for text and PDF files.
2. Add chunking and metadata persistence.
3. Add embeddings with `sentence-transformers`.
4. Store and search vectors in Qdrant.
5. Add Ollama client and RAG agent orchestration.
