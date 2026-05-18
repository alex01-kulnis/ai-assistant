from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import AgentRun, ToolCall
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatResponse, ChatSource
from app.schemas.vector import RetrievedChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.llm_service import OllamaLLMError, OllamaLLMService
from app.services.tools.order_tools import get_order_status
from app.vectorstore.qdrant_store import QdrantVectorStore, get_qdrant_vector_store

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "Ты AI support agent. Отвечай как сотрудник поддержки: "
    "понятно, спокойно и по делу. "
    "Используй только предоставленный CONTEXT. "
    "Если в CONTEXT нет ответа, честно скажи: "
    "\"В базе знаний недостаточно информации для ответа на этот вопрос.\" "
    "Не придумывай факты. "
    "Не упоминай внутренние названия файлов, chunk_id, Qdrant, "
    "embeddings или технические детали. "
    "Не пиши фразы вроде \"информация взята из файла\". "
    "Не добавляй раздел \"Источники\" в текст ответа - "
    "источники будут добавлены приложением отдельно. "
    "Если пользователь пишет на русском, отвечай на русском."
)


INTENT_KEYWORDS = {
    "refund_policy": ("возврат", "refund"),
    "payment_issue": ("оплата", "платеж", "платёж", "payment"),
    "order_status": ("заказ", "order"),
    "technical_issue": ("ошибка", "не работает", "bug"),
}


ORDER_ID_PATTERNS = (
    re.compile(r"(?:order|заказ)\s*#?\s*([a-zA-Z0-9-]{3,})", re.IGNORECASE),
    re.compile(r"#([a-zA-Z0-9-]{3,})", re.IGNORECASE),
)


@dataclass(frozen=True)
class PendingToolCall:
    tool_name: str
    input_json: dict[str, Any]
    output_json: dict[str, Any]
    status: str = "success"


class SupportAgentError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SupportAgent:
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
        started_at = time.perf_counter()
        intent = self.classify_intent(message)
        conversation: Conversation | None = None
        user_message: Message | None = None
        user_message_persisted = False
        retrieved_chunks: list[RetrievedChunk] = []
        tool_calls: list[PendingToolCall] = []

        try:
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
            user_message_persisted = True

            query_vector = self.embedding_service.embed_query(message)
            retrieved_chunks = self.vector_store.search(
                query_vector=query_vector,
                limit=self.retrieval_limit,
            )
            tool_calls = self._run_tools(intent=intent, message=message)
            llm_messages = self._build_llm_messages(
                question=message,
                intent=intent,
                retrieved_chunks=retrieved_chunks,
                tool_calls=tool_calls,
            )
            answer = await self.llm_service.generate_chat_response(llm_messages)
            answer = self._post_process_answer(
                answer=answer,
                intent=intent,
                retrieved_chunks=retrieved_chunks,
            )

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
                intent=intent,
                status="success",
                latency_ms=self._elapsed_ms(started_at),
                model_name=self.llm_service.model_name,
                retrieved_chunks_count=len(retrieved_chunks),
            )
            session.add_all([assistant_message, agent_run])
            session.add_all(self._build_tool_call_models(agent_run.id, tool_calls))
            await session.commit()

            return ChatResponse(
                conversation_id=str(conversation.id),
                message_id=str(assistant_message.id),
                answer=answer,
                sources=self._build_sources(retrieved_chunks),
            )
        except SupportAgentError:
            raise
        except OllamaLLMError as exc:
            await self._try_save_failed_agent_run(
                session=session,
                conversation_id=(
                    conversation.id if user_message_persisted and conversation else None
                ),
                user_message_id=(
                    user_message.id if user_message_persisted and user_message else None
                ),
                intent=intent,
                retrieved_chunks_count=len(retrieved_chunks),
                tool_calls=tool_calls,
                started_at=started_at,
            )
            raise SupportAgentError(f"LLM request failed: {exc}", status_code=502) from exc
        except Exception as exc:
            logger.exception(
                "Support agent failed for conversation_id=%s",
                conversation.id if conversation is not None else conversation_id,
            )
            await self._try_save_failed_agent_run(
                session=session,
                conversation_id=(
                    conversation.id if user_message_persisted and conversation else None
                ),
                user_message_id=(
                    user_message.id if user_message_persisted and user_message else None
                ),
                intent=intent,
                retrieved_chunks_count=len(retrieved_chunks),
                tool_calls=tool_calls,
                started_at=started_at,
            )
            if not user_message_persisted:
                raise SupportAgentError(
                    "Database is unavailable. Check PostgreSQL.",
                    status_code=503,
                ) from exc
            raise SupportAgentError("Failed to generate chat response.", status_code=500) from exc

    def classify_intent(self, message: str) -> str:
        normalized_message = message.casefold()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in normalized_message for keyword in keywords):
                return intent
        return "unknown"

    def _run_tools(self, intent: str, message: str) -> list[PendingToolCall]:
        if intent != "order_status":
            return []

        order_id = self._extract_order_id(message)
        if order_id is None:
            return []

        tool_input = {"order_id": order_id}
        return [
            PendingToolCall(
                tool_name="get_order_status",
                input_json=tool_input,
                output_json=get_order_status(order_id),
            )
        ]

    def _extract_order_id(self, message: str) -> str | None:
        for pattern in ORDER_ID_PATTERNS:
            match = pattern.search(message)
            if match is not None and any(character.isdigit() for character in match.group(1)):
                return match.group(1)
        return None

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
            raise SupportAgentError(
                "conversation_id must be a valid UUID.",
                status_code=400,
            ) from exc

        conversation = await session.get(Conversation, parsed_conversation_id)
        if conversation is None:
            raise SupportAgentError("Conversation not found.", status_code=404)

        return conversation

    async def _save_failed_agent_run(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        user_message_id: uuid.UUID,
        intent: str,
        retrieved_chunks_count: int,
        tool_calls: list[PendingToolCall],
        started_at: float,
    ) -> None:
        await session.rollback()
        agent_run = AgentRun(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=None,
            intent=intent,
            status="failed",
            latency_ms=self._elapsed_ms(started_at),
            model_name=self.llm_service.model_name,
            retrieved_chunks_count=retrieved_chunks_count,
        )
        session.add(agent_run)
        session.add_all(self._build_tool_call_models(agent_run.id, tool_calls))
        await session.commit()

    async def _try_save_failed_agent_run(
        self,
        *,
        session: AsyncSession,
        conversation_id: uuid.UUID | None,
        user_message_id: uuid.UUID | None,
        intent: str,
        retrieved_chunks_count: int,
        tool_calls: list[PendingToolCall],
        started_at: float,
    ) -> None:
        if conversation_id is None or user_message_id is None:
            await session.rollback()
            return

        try:
            await self._save_failed_agent_run(
                session=session,
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                intent=intent,
                retrieved_chunks_count=retrieved_chunks_count,
                tool_calls=tool_calls,
                started_at=started_at,
            )
        except Exception:
            logger.exception("Failed to save failed agent run")
            await session.rollback()

    def _build_tool_call_models(
        self,
        agent_run_id: uuid.UUID,
        tool_calls: list[PendingToolCall],
    ) -> list[ToolCall]:
        return [
            ToolCall(
                id=uuid.uuid4(),
                agent_run_id=agent_run_id,
                tool_name=tool_call.tool_name,
                input_json=tool_call.input_json,
                output_json=tool_call.output_json,
                status=tool_call.status,
            )
            for tool_call in tool_calls
        ]

    def _build_llm_messages(
        self,
        *,
        question: str,
        intent: str,
        retrieved_chunks: list[RetrievedChunk],
        tool_calls: list[PendingToolCall],
    ) -> list[dict[str, str]]:
        retrieved_context = self._build_context(retrieved_chunks)
        tool_context = self._build_tool_context(tool_calls)
        payment_instruction = ""
        if intent == "payment_issue":
            payment_instruction = (
                "\n- If context is insufficient, recommend creating a support ticket."
            )
        user_prompt = (
            f"INTENT:\n{intent}\n\n"
            f"CONTEXT:\n{retrieved_context}\n\n"
            f"TOOL RESULTS:\n{tool_context}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using only CONTEXT and TOOL RESULTS.\n"
            "- If context is insufficient, answer exactly: "
            "В базе знаний недостаточно информации для ответа на этот вопрос.\n"
            "- Be concise and helpful.\n"
            "- Do not mention filenames, chunk_id, Qdrant, embeddings, or technical details.\n"
            "- Do not add a Sources section."
            f"{payment_instruction}"
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
            context_parts.append(f"[Context fragment {index}, page={page_label}]\n{chunk.text}")

        return "\n\n".join(context_parts)

    def _build_tool_context(self, tool_calls: list[PendingToolCall]) -> str:
        if not tool_calls:
            return "No tools were called."

        return "\n".join(
            f"{tool_call.tool_name}: input={tool_call.input_json}; output={tool_call.output_json}"
            for tool_call in tool_calls
        )

    def _post_process_answer(
        self,
        *,
        answer: str,
        intent: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> str:
        if intent != "payment_issue" or not self._is_context_insufficient(retrieved_chunks):
            return answer

        recommendation = "Если вопрос с оплатой не решится, создайте обращение в поддержку."
        if recommendation.casefold() in answer.casefold():
            return answer
        return f"{answer}\n\n{recommendation}"

    def _is_context_insufficient(self, retrieved_chunks: list[RetrievedChunk]) -> bool:
        if not retrieved_chunks:
            return True
        return max(chunk.score for chunk in retrieved_chunks) < 0.35

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
