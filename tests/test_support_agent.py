from app.agents.support_agent import SupportAgent
from app.services.tools.order_tools import (
    create_support_ticket,
    get_order_status,
    get_refund_status,
)


def test_support_agent_classifies_refund_intent() -> None:
    intent = SupportAgent().classify_intent("Как оформить возврат?")

    assert intent == "refund_policy"
    assert SupportAgent().classify_intent("Need refund") == "refund_policy"


def test_support_agent_classifies_payment_intent() -> None:
    assert SupportAgent().classify_intent("Не проходит оплата") == "payment_issue"
    assert SupportAgent().classify_intent("Payment failed") == "payment_issue"


def test_support_agent_classifies_order_intent() -> None:
    assert SupportAgent().classify_intent("Где мой заказ?") == "order_status"
    assert SupportAgent().classify_intent("Order status") == "order_status"


def test_support_agent_classifies_technical_intent() -> None:
    intent = SupportAgent().classify_intent("Не работает приложение")

    assert intent == "technical_issue"
    assert SupportAgent().classify_intent("Found a bug") == "technical_issue"


def test_support_agent_classifies_unknown_intent() -> None:
    assert SupportAgent().classify_intent("Здравствуйте") == "unknown"


def test_support_agent_extracts_order_id_only_when_identifier_has_digit() -> None:
    agent = SupportAgent()

    assert agent._extract_order_id("Где заказ 12345?") == "12345"
    assert agent._extract_order_id("Order ABC-123") == "ABC-123"
    assert agent._extract_order_id("Order status") is None


def test_support_agent_runs_order_status_tool_when_order_id_is_present() -> None:
    tool_calls = SupportAgent()._run_tools(
        intent="order_status",
        message="Где заказ 12345?",
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "get_order_status"
    assert tool_calls[0].input_json == {"order_id": "12345"}
    assert tool_calls[0].output_json["order_id"] == "12345"


def test_support_agent_does_not_run_order_status_tool_without_order_id() -> None:
    tool_calls = SupportAgent()._run_tools(
        intent="order_status",
        message="Где мой заказ?",
    )

    assert tool_calls == []


def test_order_status_tool_returns_order_payload() -> None:
    payload = get_order_status("12345")

    assert payload["order_id"] == "12345"
    assert payload["status"] == "in_progress"


def test_refund_status_tool_returns_refund_payload() -> None:
    payload = get_refund_status("12345")

    assert payload["order_id"] == "12345"
    assert payload["refund_status"] == "processing"


def test_create_support_ticket_returns_deterministic_ticket() -> None:
    first_payload = create_support_ticket("payment_issue", "Не проходит оплата")
    second_payload = create_support_ticket("payment_issue", "Не проходит оплата")

    assert first_payload == second_payload
    assert first_payload["ticket_id"].startswith("TICKET-")
