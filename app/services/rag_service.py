from __future__ import annotations

import logging
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatResponse, ChatSource
from app.schemas.vector import RetrievedChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_service import OllamaLLMError, OllamaLLMService
from app.vectorstore.qdrant_store import QdrantVectorStore, get_qdrant_vector_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Ты AI support agent. "
    "Отвечай пользователю только на основе контекста. "
    "Если в контексте нет ответа, честно скажи, "
    "что информации недостаточно. "
    "Не придумывай факты. "
    "Отвечай на русском языке, "
    "если пользователь пишет на русском."
)


class RAGServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
        llm_service: OllamaLLMService | None = None,
        retrieval_limit: int = 5,
    ) -> None:
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_qdrant_vector_store()
        self.llm_service = llm_service or OllamaLLMService()
        self.retrieval_limit = retrieval_limit

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        session: AsyncSession,
    ) -> ChatResponse:
        conversation = await self._get_or_create_conversation(
            session=session,
            conversation_id=conversation_id,
        )
        user_message = Message(
            id=uuid.uuid4(),
            conversation_id=conversation.id,
            role="user",
            content=message,
        )
        session.add(user_message)
        await session.commit()

        started_at = time.perf_counter()
        retrieved_chunks: list[RetrievedChunk] = []

        try:
            query_vector = self.embedding_service.embed_query(message)
            retrieved_chunks = self.vector_store.search(
                query_vector=query_vector,
                limit=self.retrieval_limit,
            )
            llm_messages = self._build_llm_messages(
                question=message,
                retrieved_chunks=retrieved_chunks,
            )
            answer = await self.llm_service.generate_chat_response(llm_messages)

            assistant_message = Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                role="assistant",
                content=answer,
            )
            agent_run = AgentRun(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                assistant_message_id=assistant_message.id,
                status="success",
                latency_ms=self._elapsed_ms(started_at),
                model_name=self.llm_service.model_name,
                retrieved_chunks_count=len(retrieved_chunks),
            )
            session.add_all([assistant_message, agent_run])
            await session.commit()

            return ChatResponse(
                conversation_id=str(conversation.id),
                message_id=str(assistant_message.id),
                answer=answer,
                sources=self._build_sources(retrieved_chunks),
            )
        except OllamaLLMError as exc:
            await self._save_failed_agent_run(
                session=session,
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                retrieved_chunks_count=len(retrieved_chunks),
                started_at=started_at,
            )
            raise RAGServiceError(f"LLM request failed: {exc}", status_code=502) from exc
        except Exception as exc:
            logger.exception("RAG chat failed for conversation_id=%s", conversation.id)
            await self._save_failed_agent_run(
                session=session,
                conversation_id=conversation.id,
                user_message_id=user_message.id,
                retrieved_chunks_count=len(retrieved_chunks),
                started_at=started_at,
            )
            raise RAGServiceError("Failed to generate chat response.", status_code=500) from exc

    async def _get_or_create_conversation(
        self,
        *,
        session: AsyncSession,
        conversation_id: str | None,
    ) -> Conversation:
        if conversation_id is None:
            conversation = Conversation(id=uuid.uuid4())
            session.add(conversation)
            await session.flush()
            return conversation

        try:
            parsed_conversation_id = uuid.UUID(conversation_id)
        except ValueError as exc:
            raise RAGServiceError("conversation_id must be a valid UUID.", status_code=400) from exc

        conversation = await session.get(Conversation, parsed_conversation_id)
        if conversation is None:
            raise RAGServiceError("Conversation not found.", status_code=404)

        return conversation

    async def _save_failed_agent_run(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        retrieved_chunks_count: int,
        started_at: float,
    ) -> None:
        await session.rollback()
        session.add(
            AgentRun(
                id=uuid.uuid4(),
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                assistant_message_id=None,
                status="failed",
                latency_ms=self._elapsed_ms(started_at),
                model_name=self.llm_service.model_name,
                retrieved_chunks_count=retrieved_chunks_count,
            )
        )
        await session.commit()

    def _build_llm_messages(
        self,
        *,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[dict[str, str]]:
        retrieved_context = self._build_context(retrieved_chunks)
        user_prompt = (
            f"CONTEXT:\n{retrieved_context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using only CONTEXT.\n"
            "- If context is insufficient, say that there is not enough information.\n"
            "- Be concise and helpful.\n"
            "- Mention source filenames if useful."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _build_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No relevant context was found."

        context_parts: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            page_label = chunk.page_number if chunk.page_number is not None else "n/a"
            source_header = (
                f"[Source {index}] filename={chunk.filename}; "
                f"document_id={chunk.document_id}; page={page_label}; "
                f"chunk_index={chunk.chunk_index}; score={chunk.score:.4f}"
            )
            context_parts.append(f"{source_header}\n{chunk.text}")

        return "\n\n".join(context_parts)

    def _build_sources(self, retrieved_chunks: list[RetrievedChunk]) -> list[ChatSource]:
        return [
            ChatSource(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
            )
            for chunk in retrieved_chunks
        ]

    def _elapsed_ms(self, started_at: float) -> int:
        return int((time.perf_counter() - started_at) * 1000)
