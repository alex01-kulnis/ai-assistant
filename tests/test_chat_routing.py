from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.chat import get_llm_service, get_support_agent
from app.main import app
from app.schemas.chat import ChatResponse


class FakeSupportAgent:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        conversation_id: str | None,
        message: str,
        session: AsyncSession,
    ) -> ChatResponse:
        self.calls.append(
            {
                "conversation_id": conversation_id,
                "message": message,
                "session": session,
            }
        )
        return ChatResponse(
            conversation_id=conversation_id or "conversation-1",
            message_id="message-1",
            answer="RAG answer",
            sources=[],
        )


class FakeLLMService:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.messages: list[list[dict[str, str]]] = []

    async def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        self.messages.append(messages)
        return "Краткое резюме."


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_chat_endpoint_routes_rag_question_to_existing_support_agent() -> None:
    fake_agent = FakeSupportAgent()
    app.dependency_overrides[get_support_agent] = lambda: fake_agent
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={"conversation_id": "conversation-1", "question": "Как оформить возврат?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "RAG answer"
    assert payload["intent"] == "rag_question"
    assert payload["intent_confidence"] == 0.8
    assert payload["router_source"] == "rules"
    assert fake_agent.calls[0]["message"] == "Как оформить возврат?"


def test_chat_endpoint_routes_summarization_without_support_agent_call() -> None:
    fake_agent = FakeSupportAgent()
    fake_llm = FakeLLMService()
    app.dependency_overrides[get_support_agent] = lambda: fake_agent
    app.dependency_overrides[get_llm_service] = lambda: fake_llm
    client = TestClient(app)

    response = client.post(
        "/api/v1/chat",
        json={
            "question": "Суммаризируй",
            "action": "summarize",
            "selected_text": "Клиент просит возврат и уточняет сроки.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Краткое резюме."
    assert payload["sources"] == []
    assert payload["intent"] == "summarization"
    assert payload["router_source"] == "context"
    assert fake_agent.calls == []
    assert fake_llm.messages


def test_chat_endpoint_returns_controlled_fallback_for_unsupported_request() -> None:
    app.dependency_overrides[get_support_agent] = lambda: FakeSupportAgent()
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)

    response = client.post("/api/v1/chat", json={"message": "удали все документы"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "unsupported"
    assert payload["router_source"] == "rules"
    assert "не могу надежно обработать" in payload["answer"]
