# SupportOps AI Agent: Project Guide

Этот документ объясняет, что уже есть в проекте, как компоненты связаны между собой,
какой flow проходит документ при индексации и какой flow проходит пользовательский вопрос.

README остается короткой инструкцией по запуску. Этот файл стоит читать, когда нужно понять
архитектуру, точки входа и где искать конкретную логику.

## 1. Что делает проект

SupportOps AI Agent - backend для support-бота с private knowledge base.

Пользователь загружает `.txt`, `.md` или `.pdf` документы. Backend:

1. Извлекает из файла текст.
2. Режет текст на chunks.
3. Считает embeddings локальной моделью `intfloat/multilingual-e5-small`.
4. Сохраняет metadata документа и chunks в PostgreSQL.
5. Сохраняет vectors и payload chunks в Qdrant.

После этого пользователь задает вопрос через HTTP API или Telegram bot. Backend:

1. Создает или продолжает conversation.
2. Сохраняет user message.
3. Классифицирует intent простыми правилами.
4. Считает embedding вопроса.
5. Ищет top chunks в Qdrant.
6. Собирает prompt с найденным контекстом.
7. Отправляет prompt в Ollama `qwen2.5:7b`.
8. Сохраняет assistant message, agent run и tool calls.
9. Возвращает answer и sources отдельно.

LangChain намеренно не используется на MVP-этапе, чтобы архитектура RAG была видна явно.

## 2. Технологический стек

- Python 3.12
- FastAPI
- Pydantic v2
- pydantic-settings
- PostgreSQL
- SQLAlchemy 2.x async
- Alembic
- Qdrant
- Redis
- Docker Compose
- Ollama
- qwen2.5:7b
- sentence-transformers
- intfloat/multilingual-e5-small
- PyMuPDF
- aiogram 3
- httpx
- pytest
- ruff

## 3. Структура проекта

```text
app/
  api/
    routes/                 HTTP endpoints FastAPI
  agents/                   agent orchestration поверх RAG
  core/                     settings и logging
  db/                       SQLAlchemy engine, session, base
  integrations/
    telegram/               Telegram bot adapter
  models/                   SQLAlchemy ORM models
  schemas/                  Pydantic request/response schemas
  services/                 application services
  services/tools/           fake support tools
  vectorstore/              Qdrant integration
  workers/                  место под будущие background jobs

alembic/                    migrations
scripts/                    standalone scripts
tests/                      unit и integration tests
```

## 4. Куда смотреть в первую очередь

| Задача | Главный файл |
| --- | --- |
| FastAPI app и подключение routers | `app/main.py` |
| Настройки из `.env` | `app/core/config.py` |
| Логирование | `app/core/logging.py` |
| Async DB session | `app/db/session.py` |
| ORM модели документов | `app/models/document.py` |
| ORM модели диалогов | `app/models/conversation.py` |
| ORM модели agent runs и tool calls | `app/models/agent.py` |
| Upload/list documents API | `app/api/routes/documents.py` |
| Chat API | `app/api/routes/chat.py` |
| LLM health API | `app/api/routes/llm.py` |
| Индексация документа | `app/services/document_ingestion_service.py` |
| Парсинг txt/md/pdf | `app/services/document_parser.py` |
| Chunking | `app/services/chunking_service.py` |
| Embeddings | `app/services/embedding_service.py` |
| Qdrant | `app/vectorstore/qdrant_store.py` |
| Ollama client | `app/services/llm_service.py` |
| Основной support agent | `app/agents/support_agent.py` |
| Fake tools | `app/services/tools/order_tools.py` |
| Telegram bot startup | `scripts/run_telegram_bot.py` |
| Telegram handlers | `app/integrations/telegram/handlers.py` |
| Telegram backend client | `app/integrations/telegram/service.py` |
| Telegram formatting | `app/integrations/telegram/formatters.py` |

## 5. Runtime сервисы

Проект локально опирается на несколько внешних процессов.

### PostgreSQL

PostgreSQL хранит persistent metadata:

- documents
- document chunks metadata
- conversations
- messages
- agent runs
- tool calls

По умолчанию Docker публикует PostgreSQL на `localhost:5433`, чтобы не конфликтовать
с локальным PostgreSQL на `5432`.

Главная переменная:

```env
DATABASE_URL=postgresql+asyncpg://supportops:supportops@localhost:5433/supportops
```

### Qdrant

Qdrant хранит vectors и payload chunks.

Payload каждого vector point содержит:

- `document_id`
- `chunk_id`
- `chunk_index`
- `filename`
- `page_number`
- `text`

Collection:

```env
QDRANT_COLLECTION_NAME=support_knowledge_base
```

Vector size: `384`.

Distance: `cosine`.

### Redis

Redis уже поднимается через Docker Compose, но в текущем MVP пока не используется активно.
Он подготовлен под следующий этап: background indexing jobs, queues, rate limiting или cache.

### Ollama

Ollama используется как локальный LLM runtime.

Модель:

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

Backend вызывает Ollama через HTTP endpoint:

```text
POST /api/chat
```

## 6. Settings и `.env`

Все настройки читаются через `pydantic-settings` в `app/core/config.py`.

Главные переменные:

```env
APP_NAME=SupportOps AI Agent
APP_ENV=local
DEBUG=true

POSTGRES_USER=supportops
POSTGRES_PASSWORD=supportops
POSTGRES_DB=supportops
POSTGRES_HOST_PORT=5433
DATABASE_URL=postgresql+asyncpg://supportops:supportops@localhost:5433/supportops

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=support_knowledge_base
REDIS_URL=redis://localhost:6379/0

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small

TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USER_IDS=
TELEGRAM_USE_BACKEND_HTTP=true
TELEGRAM_BACKEND_CHAT_URL=http://localhost:8000/api/v1/chat
```

Важно: реальный Telegram token хранится только в локальном `.env` и не должен попадать в git.

## 7. Database schema

### `documents`

Файл, загруженный в knowledge base.

Поля:

- `id`
- `filename`
- `content_type`
- `status`
- `created_at`
- `updated_at`

Статусы сейчас:

- `processing`
- `indexed`
- `failed`

### `document_chunks`

Metadata chunks, которые получились из документа.

Поля:

- `id`
- `document_id`
- `chunk_index`
- `text`
- `page_number`
- `qdrant_point_id`
- `created_at`

Сам vector хранится не в PostgreSQL, а в Qdrant. PostgreSQL хранит связь между документом,
chunk и Qdrant point.

### `conversations`

Диалог пользователя с агентом.

Поля:

- `id`
- `user_id`
- `created_at`
- `updated_at`

В MVP `user_id` nullable. Telegram conversation state пока хранится in-memory внутри bot process.

### `messages`

Сообщения внутри conversation.

Поля:

- `id`
- `conversation_id`
- `role`
- `content`
- `created_at`

`role` ограничен значениями:

- `user`
- `assistant`
- `system`

### `agent_runs`

Одна попытка agent execution для пользовательского сообщения.

Поля:

- `id`
- `conversation_id`
- `user_message_id`
- `assistant_message_id`
- `intent`
- `status`
- `latency_ms`
- `model_name`
- `retrieved_chunks_count`
- `created_at`

Эта таблица нужна для наблюдаемости: можно смотреть, как часто агент падает, сколько chunks
он достает, сколько времени занимает ответ, какой intent был определен.

### `tool_calls`

Вызовы tools внутри agent run.

Поля:

- `id`
- `agent_run_id`
- `tool_name`
- `input_json`
- `output_json`
- `status`
- `created_at`

Сейчас tools fake, но схема уже похожа на production-подход.

## 8. Flow загрузки документа

Endpoint:

```text
POST /api/v1/documents/upload
```

Route:

```text
app/api/routes/documents.py
```

Основная логика:

```text
app/services/document_ingestion_service.py
```

Пошагово:

1. FastAPI принимает `UploadFile`.
2. Route читает bytes файла.
3. `DocumentIngestionService.index_uploaded_document()` создает запись `Document`
   со статусом `processing`.
4. Файл временно сохраняется в temp directory.
5. `DocumentParser` определяет расширение:
   - `.txt`
   - `.md`
   - `.pdf`
6. Для `.txt` и `.md` файл читается как UTF-8.
7. Для `.pdf` используется PyMuPDF, текст извлекается постранично.
8. Если PDF не содержит текстовый слой, возвращается ошибка:
   `PDF does not contain extractable text. OCR is not supported yet.`
9. `TextChunkingService.split_pages()` режет текст на chunks.
10. `EmbeddingService.embed_documents()` считает embeddings.
11. Для E5-модели каждый документный chunk получает prefix:
    `passage: `
12. `QdrantVectorStore.upsert_chunks()` создает point ids и пишет vectors в Qdrant.
13. PostgreSQL сохраняет `DocumentChunk` metadata и `qdrant_point_id`.
14. `Document.status` меняется на `indexed`.
15. API возвращает:

```json
{
  "document_id": "...",
  "filename": "refund_policy.txt",
  "chunks_count": 12,
  "status": "indexed"
}
```

Если ошибка возникает на этапе parsing/chunking/embedding/Qdrant, документ помечается
как `failed`.

## 9. Flow списка документов

Endpoint:

```text
GET /api/v1/documents
```

Route:

```text
app/api/routes/documents.py
```

Service:

```text
DocumentIngestionService.list_documents()
```

Что делает:

1. Читает `documents`.
2. Делает `outer join` на `document_chunks`.
3. Считает `chunks_count`.
4. Возвращает список документов, отсортированный по `created_at desc`.

Response item:

```json
{
  "id": "...",
  "filename": "refund_policy.txt",
  "content_type": "text/plain",
  "status": "indexed",
  "chunks_count": 1,
  "created_at": "..."
}
```

## 10. Flow chat request

Endpoint:

```text
POST /api/v1/chat
```

Route:

```text
app/api/routes/chat.py
```

Главная логика:

```text
app/agents/support_agent.py
```

Request:

```json
{
  "conversation_id": null,
  "message": "Как оформить возврат?"
}
```

Пошагово:

1. `ChatRequest` валидирует, что `message` не пустой.
2. Route вызывает `SupportAgent.chat()`.
3. Если `conversation_id` не передан, создается новая `Conversation`.
4. Если `conversation_id` передан, агент ищет existing conversation.
5. User message сохраняется в `messages`.
6. Агент классифицирует intent keyword-based правилами.
7. `EmbeddingService.embed_query()` считает vector вопроса.
8. Для E5-модели user query получает prefix:
   `query: `
9. `QdrantVectorStore.search()` ищет top 5 chunks.
10. Агент проверяет, нужно ли вызвать fake tool.
11. Агент собирает prompt:
    - system prompt
    - intent
    - retrieved context
    - tool results
    - user question
    - instructions
12. `OllamaLLMService.generate_chat_response()` отправляет prompt в Ollama.
13. Assistant answer сохраняется в `messages`.
14. Создается `AgentRun` со статусом `success`.
15. Если были tools, сохраняются `ToolCall`.
16. API возвращает answer и sources отдельно.

Response:

```json
{
  "conversation_id": "...",
  "message_id": "...",
  "answer": "Для оформления возврата ...",
  "sources": [
    {
      "document_id": "...",
      "filename": "refund_policy.txt",
      "page_number": null,
      "chunk_index": 0,
      "score": 0.899
    }
  ]
}
```

## 11. Intent classification

Intent classification находится в:

```text
app/agents/support_agent.py
```

Сейчас это простые keyword rules:

| Intent | Keywords |
| --- | --- |
| `refund_policy` | `возврат`, `refund` |
| `payment_issue` | `оплата`, `платеж`, `платёж`, `payment` |
| `order_status` | `заказ`, `order` |
| `technical_issue` | `ошибка`, `не работает`, `bug` |
| `unknown` | fallback |

Это простой MVP-вариант. Следующий production-like шаг - добавить LLM fallback или отдельную
маленькую intent classification модель.

## 12. Tools flow

Fake tools лежат здесь:

```text
app/services/tools/order_tools.py
```

Сейчас есть:

- `get_order_status(order_id: str)`
- `get_refund_status(order_id: str)`
- `create_support_ticket(reason: str, user_message: str)`

В текущем agent flow реально вызывается `get_order_status`, если:

1. Intent равен `order_status`.
2. В сообщении найден order id.

Результат tool добавляется в prompt как `TOOL RESULTS` и сохраняется в PostgreSQL как `ToolCall`.

## 13. Prompting и ограничения ответа

System prompt находится в:

```text
app/agents/support_agent.py
```

Главная идея prompt:

- отвечать как сотрудник поддержки;
- использовать только retrieved context и tool results;
- не придумывать факты;
- не упоминать filenames, chunk ids, Qdrant, embeddings и внутренние детали;
- не добавлять раздел `Источники` в answer;
- если данных нет, отвечать:
  `В базе знаний недостаточно информации для ответа на этот вопрос.`

Sources возвращаются отдельно в API response. Это важно: LLM не должна сама оформлять источники,
потому что приложение уже знает exact metadata найденных chunks.

## 14. Telegram bot flow

Telegram bot - отдельный процесс. Он не стартует внутри FastAPI.

Startup script:

```text
scripts/run_telegram_bot.py
```

Bot setup:

```text
app/integrations/telegram/bot.py
```

Handlers:

```text
app/integrations/telegram/handlers.py
```

Telegram chat client:

```text
app/integrations/telegram/service.py
```

Formatting:

```text
app/integrations/telegram/formatters.py
```

### Commands

- `/start` - приветствие и краткое описание.
- `/help` - команды и примеры вопросов.
- `/new` - сбросить текущий conversation для Telegram user.
- `/sources` - показать sources последнего ответа.

### Access control

Переменная:

```env
TELEGRAM_ALLOWED_USER_IDS=123456,789012
```

Если переменная пустая, бот доступен всем.

Если задан список id через запятую, доступ есть только у этих Telegram users.

### Conversation state

В MVP состояние Telegram хранится in-memory:

```text
telegram_user_id -> conversation_id
telegram_user_id -> last_sources
```

Это находится в `TelegramSessionStore`.

Важно: при перезапуске bot process это состояние теряется. Для production-like следующего шага
его можно перенести в PostgreSQL.

### Backend integration modes

Есть два режима:

```env
TELEGRAM_USE_BACKEND_HTTP=true
```

В этом режиме bot отправляет HTTP request в FastAPI:

```text
POST http://localhost:8000/api/v1/chat
```

Это самый стабильный режим для локальной разработки, потому что FastAPI и Telegram bot
запущены как два отдельных процесса.

```env
TELEGRAM_USE_BACKEND_HTTP=false
```

В этом режиме bot вызывает `SupportAgent` напрямую внутри bot process. Режим полезен для
экспериментов, но для debugging проще HTTP mode.

### Telegram answer formatting

Bot получает от backend:

- `answer`
- `sources`

Потом formatter:

1. Очищает случайные source mentions из LLM answer.
2. Отправляет чистый answer.
3. Если sources есть, добавляет короткую подсказку:

```text
Источник: refund_policy.txt
Подробнее: /sources
```

Команда `/sources` показывает подробности:

```text
1. refund_policy.txt | page: - | chunk: 0 | score: 0.887
```

## 15. API endpoints

### Root

```text
GET /
```

Возвращает имя приложения.

### Health

```text
GET /health
```

Быстрая проверка, что FastAPI поднялся.

### LLM health

```text
GET /api/v1/llm/health
```

Проверяет, отвечает ли Ollama.

### Upload document

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@refund_policy.txt"
```

### List documents

```bash
curl http://127.0.0.1:8000/api/v1/documents
```

### Chat

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Как оформить возврат?"}'
```

Продолжение существующего диалога:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"<conversation-id>","message":"А сколько ждать деньги?"}'
```

## 16. Как запустить локально

### 1. Установить зависимости

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Подготовить `.env`

```bash
cp .env.example .env
```

Проверь, что `DATABASE_URL` смотрит на `localhost:5433`, если используешь текущий Docker Compose.

### 3. Запустить инфраструктуру

```bash
make docker-up
```

Проверка Qdrant dashboard:

```text
http://localhost:6333/dashboard
```

### 4. Применить миграции

```bash
make migrate
```

### 5. Подготовить Ollama

```bash
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

### 6. Запустить FastAPI

```bash
make dev
```

OpenAPI:

```text
http://127.0.0.1:8000/docs
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

### 7. Запустить Telegram bot

В `.env`:

```env
TELEGRAM_BOT_TOKEN=<token-from-botfather>
TELEGRAM_USE_BACKEND_HTTP=true
TELEGRAM_BACKEND_CHAT_URL=http://localhost:8000/api/v1/chat
```

Запуск:

```bash
make telegram-bot
```

## 17. Tests

Запуск всех тестов:

```bash
make test
```

Линт:

```bash
make lint
```

Что покрыто:

- health endpoint;
- document parser;
- chunking;
- embedding service;
- Qdrant upsert/search integration;
- Ollama service error handling;
- support agent behavior;
- Telegram access control, formatting и HTTP client payload.

## 18. Типичные проблемы и куда смотреть

### `Connection refused` к PostgreSQL

Проверь:

```bash
docker compose ps
```

Проверь порт:

```env
DATABASE_URL=postgresql+asyncpg://supportops:supportops@localhost:5433/supportops
```

Если Docker публикует PostgreSQL на `5433`, а app смотрит на `5432`, chat/upload будут падать.

### `role "supportops" does not exist`

Обычно это значит, что app подключился не к Docker PostgreSQL, а к локальному PostgreSQL
на другом порту.

Проверь `DATABASE_URL` и `POSTGRES_HOST_PORT`.

### Qdrant недоступен

Проверь:

```text
http://localhost:6333/dashboard
```

И:

```bash
docker compose ps
```

### Ollama недоступна

Проверь:

```bash
curl http://localhost:11434/api/tags
```

И:

```bash
ollama run qwen2.5:7b
```

Через backend:

```text
GET /api/v1/llm/health
```

### Telegram bot отвечает, что backend недоступен

Проверь:

```bash
make dev
curl http://127.0.0.1:8000/health
```

И переменные:

```env
TELEGRAM_USE_BACKEND_HTTP=true
TELEGRAM_BACKEND_CHAT_URL=http://localhost:8000/api/v1/chat
```

### Telegram bot отвечает, что LLM недоступна

FastAPI работает, но Ollama не отвечает или модель не загружена.

Проверь:

```bash
ollama pull qwen2.5:7b
ollama run qwen2.5:7b
```

## 19. Что сейчас production-like, а что еще MVP

### Уже production-like

- слои разделены: routes, services, agents, db, models, schemas, vectorstore, integrations;
- route handlers тонкие;
- settings через `.env` и pydantic-settings;
- async FastAPI и async SQLAlchemy session;
- Alembic migrations;
- PostgreSQL хранит аудит диалогов и agent runs;
- Qdrant хранит vectors отдельно от relational metadata;
- LLM вызывается через отдельный service;
- есть tests и lint;
- Telegram bot вынесен в отдельный процесс.

### Пока MVP

- indexing синхронный, не через worker;
- Redis пока не используется;
- intent classification keyword-based;
- tools fake;
- Telegram session store in-memory;
- нет auth для HTTP API;
- нет delete/reindex document flow;
- нет OCR для PDF без текстового слоя;
- нет observability stack: metrics/tracing/dashboard;
- нет reranking;
- нет hybrid search.

## 20. Логика расширения проекта

Ближайшие улучшения, которые логично делать дальше:

1. Вынести document indexing в background worker.
2. Использовать Redis queue или другой broker.
3. Добавить endpoint удаления документа и чистку Qdrant points.
4. Добавить reindex flow.
5. Хранить Telegram session state в PostgreSQL.
6. Добавить auth/API keys для backend endpoints.
7. Добавить LLM-based intent fallback.
8. Добавить reranking retrieved chunks.
9. Добавить streaming ответов.
10. Добавить admin endpoints для просмотра conversations, messages и agent runs.

## 21. Быстрая mental model

Если коротко, проект состоит из двух больших pipelines.

### Knowledge base pipeline

```text
UploadFile
  -> DocumentParser
  -> TextChunkingService
  -> EmbeddingService
  -> QdrantVectorStore
  -> PostgreSQL metadata
```

### Question answering pipeline

```text
User question
  -> SupportAgent
  -> intent classification
  -> EmbeddingService
  -> QdrantVectorStore.search
  -> prompt with context
  -> OllamaLLMService
  -> PostgreSQL messages/agent_run/tool_calls
  -> ChatResponse(answer, sources)
```

### Telegram pipeline

```text
Telegram message
  -> aiogram handler
  -> TelegramChatClient
  -> FastAPI /api/v1/chat
  -> SupportAgent
  -> formatted Telegram answer
```

Это главное, что нужно держать в голове при чтении кода.
