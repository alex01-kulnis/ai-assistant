# SupportOps AI Agent

SupportOps AI Agent is a production-like Python backend pet project for automated support
question answering over a private knowledge base.

Users upload `.txt`, `.md`, or `.pdf` documents. The system parses them, chunks the text,
creates local embeddings, stores vectors in Qdrant, and answers questions through a local
Ollama LLM using retrieved context.

The MVP intentionally avoids LangChain so the core RAG and agent architecture stays explicit.

## Features

- Async FastAPI application.
- PostgreSQL metadata storage with SQLAlchemy 2.x and Alembic.
- Qdrant vector search with cosine similarity.
- Redis prepared for future background workers.
- Local embeddings via `sentence-transformers`.
- PDF text extraction via PyMuPDF.
- Local LLM calls through Ollama `qwen2.5:7b`.
- Document upload and synchronous indexing.
- RAG chat endpoint with sources.
- Simple support agent layer with keyword intent classification.
- Fake tools with persisted `ToolCall` records.
- Basic unit and integration test structure.

## Stack

- Python 3.12
- FastAPI
- PostgreSQL
- SQLAlchemy 2.x
- Alembic
- Pydantic v2
- Qdrant
- Redis
- Docker Compose
- Ollama
- `sentence-transformers`
- PyMuPDF
- httpx
- pytest
- ruff

## Architecture

```text
app/
  api/routes/        FastAPI routes
  agents/            Support agent orchestration
  core/              settings and logging
  db/                SQLAlchemy engine/session/base
  models/            PostgreSQL ORM models
  schemas/           Pydantic request/response schemas
  services/          parser, chunker, embeddings, LLM, RAG, ingestion
  services/tools/    fake support tools
  vectorstore/       Qdrant integration
  workers/           reserved for background jobs
tests/               unit and integration tests
```

Main layers:

- `api`: thin HTTP layer, no business logic.
- `services`: application services for document ingestion, embeddings, LLM, and RAG.
- `agents`: support-specific orchestration above RAG.
- `models`: persisted domain entities.
- `vectorstore`: direct Qdrant client integration.

## Flow

### Document Indexing

```mermaid
flowchart LR
    A["Upload document"] --> B["DocumentParser"]
    B --> C["TextChunkingService"]
    C --> D["EmbeddingService"]
    D --> E["QdrantVectorStore"]
    C --> F["PostgreSQL metadata"]
    E --> G["Indexed knowledge base"]
```

### Chat

```mermaid
flowchart LR
    A["POST /api/v1/chat"] --> B["SupportAgent"]
    B --> C["Intent classification"]
    C --> D["EmbeddingService"]
    D --> E["Qdrant search"]
    C --> F["Optional fake tool"]
    E --> G["Prompt context"]
    F --> G
    G --> H["Ollama qwen2.5:7b"]
    H --> I["Answer + sources"]
    B --> J["Conversation, Message, AgentRun, ToolCall"]
```

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create local environment config:

```bash
cp .env.example .env
```

Important defaults:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
QDRANT_COLLECTION_NAME=support_knowledge_base
```

## Docker

Start PostgreSQL, Qdrant, and Redis:

```bash
make docker-up
```

Equivalent command:

```bash
docker compose up -d
```

Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

Stop infrastructure:

```bash
make docker-down
```

## Database Migrations

Apply migrations:

```bash
make migrate
```

Equivalent command:

```bash
alembic upgrade head
```

## Ollama

Install the model:

```bash
ollama pull qwen2.5:7b
```

Run the model:

```bash
ollama run qwen2.5:7b
```

Check Ollama directly:

```bash
curl http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5:7b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": false
  }'
```

Check Ollama through FastAPI:

```bash
curl http://localhost:8000/api/v1/llm/health
```

## Run FastAPI

```bash
make dev
```

Equivalent command:

```bash
uvicorn app.main:app --reload
```

Useful URLs:

- API root: http://localhost:8000/
- Health check: http://localhost:8000/health
- OpenAPI docs: http://localhost:8000/docs
- LLM health: http://localhost:8000/api/v1/llm/health

## Upload Document

Create a sample document:

```bash
cat > example.txt <<'EOF'
Возврат можно оформить в течение 14 дней после получения заказа.
Для возврата нужен номер заказа и причина обращения.
EOF
```

Upload it into the knowledge base:

Endpoint: `POST /api/v1/documents/upload`

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@./example.txt"
```

Example response:

```json
{
  "document_id": "7ed15b28-91ef-471f-a74d-7f509f9c741d",
  "filename": "example.txt",
  "chunks_count": 1,
  "status": "indexed"
}
```

List documents:

```bash
curl http://localhost:8000/api/v1/documents
```

## Chat Request

Ask a question:

Endpoint: `POST /api/v1/chat`

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Как оформить возврат?"}'
```

Continue an existing conversation:

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"<conversation-id>","message":"Какие сроки возврата?"}'
```

Example response:

```json
{
  "conversation_id": "2af9d5b0-a240-4e30-90cc-2eaf82b3d2df",
  "message_id": "b31d11c2-f60a-4e60-9d53-18f67d3f1110",
  "answer": "Возврат можно оформить в течение 14 дней после получения заказа.",
  "sources": [
    {
      "document_id": "7ed15b28-91ef-471f-a74d-7f509f9c741d",
      "filename": "example.txt",
      "page_number": null,
      "chunk_index": 0,
      "score": 0.82
    }
  ]
}
```

## Tests

Run unit tests:

```bash
make test
```

Equivalent command:

```bash
pytest
```

Run lint:

```bash
make lint
```

Run Qdrant integration tests after starting Qdrant:

```bash
make docker-up
pytest -m integration
```

## Makefile Commands

```bash
make dev          # run FastAPI with reload
make docker-up    # start PostgreSQL, Qdrant, Redis
make docker-down  # stop Docker Compose services
make migrate      # apply Alembic migrations
make test         # run pytest
make lint         # run ruff
```

## Architecture Decisions

### Why Qdrant

Qdrant is a focused vector database with a straightforward Python client, persistent storage,
payload filtering, and good local Docker support. It keeps vector search separate from relational
metadata and matches the RAG use case cleanly.

### Why PostgreSQL

PostgreSQL stores source-of-truth application data: documents, chunks metadata, conversations,
messages, agent runs, and tool calls. This data is relational, auditable, and should survive
vector index rebuilds.

### Why Ollama

Ollama makes the LLM fully local for development. It avoids external API keys, keeps the project
easy to demo offline, and allows experimentation with `qwen2.5:7b` through a simple HTTP API.

### Why No LangChain In The MVP

The MVP avoids LangChain to make the architecture explicit: parsing, chunking, embedding,
retrieval, prompting, tool calls, and persistence are implemented directly. This makes the project
better for learning and easier to debug. A framework can be introduced later if it removes real
complexity.

## Roadmap

1. Move document indexing to background workers.
2. Add Redis-backed job orchestration.
3. Add auth and user-scoped knowledge bases.
4. Add richer integration tests for the full ingestion and chat flow.
5. Add streaming chat responses.
6. Add OCR for scanned PDFs.
7. Add observability for latency, retrieval scores, and LLM failures.
