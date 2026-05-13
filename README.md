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
- document parsing and text chunking services
- embedding service based on `sentence-transformers`
- Qdrant vector store integration
- document upload API for synchronous indexing
- Ollama LLM service and health endpoint
- support agent layer with keyword intent classification and fake tools
- basic pytest coverage for root and health endpoints

Indexing is synchronous for now. It will move to background workers later.

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

## Ollama

Download the local LLM model:

```bash
ollama pull qwen2.5:7b
```

Run the model:

```bash
ollama run qwen2.5:7b
```

Check that Ollama works directly:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }'
```

Check Ollama from the FastAPI app:

```bash
curl http://localhost:8000/api/v1/llm/health
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

## Documents API

Upload a document into the knowledge base:

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@./example.txt"
```

List indexed documents:

```bash
curl http://localhost:8000/api/v1/documents
```

## Chat API

Ask a question against the indexed knowledge base:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Как оформить возврат?"}'
```

Continue an existing conversation:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"<conversation-id>","message":"Какие сроки?"}'
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

1. Move document indexing to background workers.
2. Add Redis-backed job orchestration.
3. Persist detailed tool calls for retrieval and LLM calls.
4. Add richer integration tests for the full ingestion and RAG flow.
5. Add auth and user-scoped knowledge bases.
