from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agents.langchain_support_agent import (
    LangChainSupportAgent,
    get_langchain_support_agent,
)
from app.agents.support_agent import SupportAgentError
from app.main import app
from app.models.agent import AgentRun
from app.models.conversation import Conversation, Message
from app.schemas.chat import ChatResponse, ChatSource
from app.schemas.vector import RetrievedChunk
from app.services.langchain_rag_service import (
    LangChainRAGGenerationError,
    LangChainRAGResult,
    LangChainRAGService,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any]


class FakeVectorStore:
    def __init__(self) -> None:
        self.queries: list[tuple[str, int]] = []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[FakeDocument, float]]:
        self.queries.append((query, k))
        return [
            (
                FakeDocument(
                    page_content="Возврат оформляется через форму поддержки.",
                    metadata={
                        "point_id": "point-1",
                        "document_id": "document-1",
                        "chunk_id": "chunk-1",
                        "filename": "refund_policy.txt",
                        "page_number": None,
                        "chunk_index": 0,
                    },
                ),
                0.887,
            )
        ]


class FakeLLM:
    def __init__(self, content: str = "Заполните форму поддержки.") -> None:
        self.content = content
        self.messages: Any = None

    async def ainvoke(self, input: Any) -> Any:
        self.messages = input
        return type("FakeAIMessage", (), {"content": self.content})()


class FakeRAGService:
    model_name = "fake-langchain-llm"

    async def answer(self, *, question: str, intent: str, tool_results: str) -> LangChainRAGResult:
        return LangChainRAGResult(
            answer=f"Ответ на вопрос: {question}",
            retrieved_chunks=[
                RetrievedChunk(
                    point_id="point-1",
                    score=0.887,
                    text="Возврат оформляется через форму поддержки.",
                    document_id="document-1",
                    chunk_id="chunk-1",
                    filename="refund_policy.txt",
                    page_number=None,
                    chunk_index=0,
                )
            ],
        )


class FailingRAGService:
    model_name = "fake-langchain-llm"

    async def answer(self, *, question: str, intent: str, tool_results: str) -> LangChainRAGResult:
        raise LangChainRAGGenerationError("Ollama failed")


class FakeAsyncSession:
    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commits = 0
        self.rollbacks = 0
        self.flushes = 0

    def add(self, instance: Any) -> None:
        self.added.append(instance)

    def add_all(self, instances: list[Any]) -> None:
        self.added.extend(instances)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def flush(self) -> None:
        self.flushes += 1

    async def get(self, model: type, primary_key: Any) -> Any:
        return None


class FakeEndpointAgent:
    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        session: Any,
    ) -> ChatResponse:
        return ChatResponse(
            conversation_id=str(uuid.uuid4()),
            message_id=str(uuid.uuid4()),
            answer=f"LangChain answer: {message}",
            sources=[
                ChatSource(
                    document_id="document-1",
                    filename="refund_policy.txt",
                    page_number=None,
                    chunk_index=0,
                    score=0.887,
                )
            ],
        )


@pytest.mark.asyncio
async def test_langchain_rag_service_uses_mocked_retriever_and_llm() -> None:
    vector_store = FakeVectorStore()
    llm = FakeLLM()
    service = LangChainRAGService(
        vector_store=vector_store,
        llm=llm,
        retrieval_limit=5,
    )

    result = await service.answer(
        question="Как оформить возврат?",
        intent="refund_policy",
        tool_results="No tools were called.",
    )

    assert result.answer == "Заполните форму поддержки."
    assert vector_store.queries == [("Как оформить возврат?", 5)]
    assert result.retrieved_chunks[0].filename == "refund_policy.txt"
    assert result.retrieved_chunks[0].score == 0.887
    assert llm.messages is not None


def test_chat_langchain_endpoint_returns_chat_response_shape() -> None:
    app.dependency_overrides[get_langchain_support_agent] = lambda: FakeEndpointAgent()
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/chat/langchain",
            json={"message": "Как оформить возврат?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    response_data = response.json()
    assert set(response_data) == {"conversation_id", "message_id", "answer", "sources"}
    assert response_data["answer"] == "LangChain answer: Как оформить возврат?"
    assert response_data["sources"][0]["filename"] == "refund_policy.txt"


@pytest.mark.asyncio
async def test_langchain_support_agent_saves_messages_agent_run_and_sources() -> None:
    session = FakeAsyncSession()
    agent = LangChainSupportAgent(rag_service=FakeRAGService())  # type: ignore[arg-type]

    response = await agent.chat(
        conversation_id=None,
        message="Как оформить возврат?",
        session=session,  # type: ignore[arg-type]
    )

    messages = [item for item in session.added if isinstance(item, Message)]
    agent_runs = [item for item in session.added if isinstance(item, AgentRun)]

    assert [message.role for message in messages] == ["user", "assistant"]
    assert len(agent_runs) == 1
    assert agent_runs[0].status == "success"
    assert agent_runs[0].retrieved_chunks_count == 1
    assert response.sources[0].filename == "refund_policy.txt"
    assert response.sources[0].score == 0.887


@pytest.mark.asyncio
async def test_langchain_support_agent_saves_failed_run_for_llm_error() -> None:
    session = FakeAsyncSession()
    agent = LangChainSupportAgent(rag_service=FailingRAGService())  # type: ignore[arg-type]

    with pytest.raises(SupportAgentError) as exc_info:
        await agent.chat(
            conversation_id=None,
            message="Как оформить возврат?",
            session=session,  # type: ignore[arg-type]
        )

    agent_runs = [item for item in session.added if isinstance(item, AgentRun)]
    assert exc_info.value.status_code == 502
    assert len(agent_runs) == 1
    assert agent_runs[0].status == "failed"


@pytest.mark.asyncio
async def test_langchain_support_agent_continues_existing_conversation() -> None:
    conversation = Conversation(id=uuid.uuid4())

    class ExistingConversationSession(FakeAsyncSession):
        async def get(self, model: type, primary_key: Any) -> Any:
            return conversation

    session = ExistingConversationSession()
    agent = LangChainSupportAgent(rag_service=FakeRAGService())  # type: ignore[arg-type]

    response = await agent.chat(
        conversation_id=str(conversation.id),
        message="Как оформить возврат?",
        session=session,  # type: ignore[arg-type]
    )

    assert response.conversation_id == str(conversation.id)
