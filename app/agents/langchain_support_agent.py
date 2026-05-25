from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import (
    INTENT_KEYWORDS,
    ORDER_ID_PATTERNS,
    PendingToolCall,
    SupportAgentError,
)
from app.models.agent import AgentRun, ToolCall
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatResponse, ChatSource
from app.schemas.vector import RetrievedChunk
from app.services.langchain_rag_service import (
    LangChainRAGError,
    LangChainRAGGenerationError,
    LangChainRAGResult,
    LangChainRAGService,
    get_langchain_rag_service,
)
from app.services.tools.order_tools import get_order_status

logger = logging.getLogger(__name__)


class LangChainSupportAgent:
    def __init__(
        self,
        *,
        rag_service: LangChainRAGService | None = None,
    ) -> None:
        self.rag_service = rag_service or get_langchain_rag_service()

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

            tool_calls = self._run_tools(intent=intent, message=message)
            rag_result = await self.rag_service.answer(
                question=message,
                intent=intent,
                tool_results=self._build_tool_context(tool_calls),
            )
            retrieved_chunks = rag_result.retrieved_chunks
            answer = self._post_process_answer(
                rag_result=rag_result,
                intent=intent,
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
                model_name=self.rag_service.model_name,
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
        except LangChainRAGGenerationError as exc:
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
        except LangChainRAGError as exc:
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
            raise SupportAgentError("LangChain RAG request failed.", status_code=500) from exc
        except Exception as exc:
            logger.exception(
                "LangChain support agent failed for conversation_id=%s",
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
            model_name=self.rag_service.model_name,
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
            logger.exception("Failed to save failed LangChain agent run")
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

    def _build_tool_context(self, tool_calls: list[PendingToolCall]) -> str:
        if not tool_calls:
            return "No tools were called."

        return "\n".join(
            f"{tool_call.tool_name}: input={tool_call.input_json}; output={tool_call.output_json}"
            for tool_call in tool_calls
        )

    def _post_process_answer(self, *, rag_result: LangChainRAGResult, intent: str) -> str:
        answer = rag_result.answer
        if intent != "payment_issue" or not self._is_context_insufficient(
            rag_result.retrieved_chunks
        ):
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


@lru_cache
def get_langchain_support_agent() -> LangChainSupportAgent:
    return LangChainSupportAgent()
