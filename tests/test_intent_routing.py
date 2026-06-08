from __future__ import annotations

import pytest

from app.routing.classifier import classify_intent
from app.routing.context_router import route_by_context
from app.routing.rule_router import route_by_rules
from app.routing.schemas import IntentResult
from app.schemas.chat import ChatRequest


def test_context_router_routes_summarization_from_action_and_selected_text() -> None:
    result = route_by_context(
        ChatRequest(
            question="Сделай кратко",
            action="summarize",
            selected_text="Длинный текст обращения клиента.",
        )
    )

    assert result == IntentResult(intent="summarization", confidence=0.99, source="context")


def test_context_router_routes_customer_analysis_from_customer_id() -> None:
    result = route_by_context(
        ChatRequest(question="Что предложить клиенту?", customer_id="cust-123")
    )

    assert result == IntentResult(intent="customer_analysis", confidence=0.95, source="context")


def test_context_router_routes_rag_from_document_id() -> None:
    result = route_by_context(ChatRequest(question="Что в документе?", document_id="doc-123"))

    assert result == IntentResult(intent="rag_question", confidence=0.95, source="context")


def test_context_router_returns_none_without_context() -> None:
    assert route_by_context(ChatRequest(question="Здравствуйте")) is None


@pytest.mark.parametrize(
    ("question", "expected_intent", "expected_confidence"),
    [
        ("удали все документы", "unsupported", 0.95),
        ("Суммаризируй это обращение", "summarization", 0.90),
        ("Оцени churn risk клиента", "customer_analysis", 0.85),
        ("Как оформить возврат?", "rag_question", 0.80),
    ],
)
def test_rule_router_routes_known_keywords(
    question: str,
    expected_intent: str,
    expected_confidence: float,
) -> None:
    result = route_by_rules(question)

    assert result is not None
    assert result.intent == expected_intent
    assert result.confidence == expected_confidence
    assert result.source == "rules"


def test_rule_router_returns_none_for_unknown_question() -> None:
    assert route_by_rules("Здравствуйте") is None


def test_rule_router_keeps_document_question_with_customer_word_as_rag() -> None:
    result = route_by_rules("Что делать если клиент просит SLA?")

    assert result is not None
    assert result.intent == "rag_question"


def test_rule_router_routes_strong_customer_signal_before_rag_keyword() -> None:
    result = route_by_rules("Что делать если у клиента высокий churn risk?")

    assert result is not None
    assert result.intent == "customer_analysis"


@pytest.mark.asyncio
async def test_classifier_prioritizes_context_over_rules() -> None:
    result = await classify_intent(
        ChatRequest(
            question="удали все документы",
            document_id="doc-123",
        )
    )

    assert result.intent == "rag_question"
    assert result.source == "context"


@pytest.mark.asyncio
async def test_classifier_falls_back_when_llm_confidence_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_by_llm(*args: object, **kwargs: object) -> IntentResult:
        return IntentResult(intent="rag_question", confidence=0.20, source="llm")

    monkeypatch.setattr("app.routing.classifier.route_by_llm", fake_route_by_llm)

    result = await classify_intent(ChatRequest(question="Здравствуйте"))

    assert result.intent == "unsupported"
    assert result.confidence == 0.0
    assert result.source == "fallback"


@pytest.mark.asyncio
async def test_classifier_uses_mocked_llm_when_rules_do_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_by_llm(*args: object, **kwargs: object) -> IntentResult:
        return IntentResult(intent="customer_analysis", confidence=0.72, source="llm")

    monkeypatch.setattr("app.routing.classifier.route_by_llm", fake_route_by_llm)

    result = await classify_intent(ChatRequest(question="Оцени профиль пользователя"))

    assert result.intent == "customer_analysis"
    assert result.confidence == 0.72
    assert result.source == "llm"
