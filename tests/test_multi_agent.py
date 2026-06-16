from __future__ import annotations

import pytest

from app.agents.base import BaseAgent
from app.agents.customer_analysis_agent import CustomerAnalysisAgent
from app.agents.guardrails import validate_agent_result
from app.agents.state import AgentState
from app.agents.supervisor import SupervisorAgent


class FakeAgent(BaseAgent):
    def __init__(self, name: str, *, with_sources: bool = True) -> None:
        self.name = name
        self.calls = 0
        self.with_sources = with_sources

    async def run(self, state: AgentState) -> AgentState:
        self.calls += 1
        state.selected_agent = self.name
        state.current_agent = self.name
        state.answer = f"{self.name} answer"
        if self.with_sources:
            state.sources = [
                {
                    "document_id": "doc-1",
                    "filename": "doc.txt",
                    "page_number": None,
                    "chunk_index": 0,
                    "score": 0.9,
                }
            ]
        state.add_trace_step(self.name, "fake_completed")
        return state


class FakeLLMService:
    model_name = "fake-llm"

    async def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        return (
            "1. Краткий статус клиента: premium.\n"
            "2. Риск / сигналы: высокий rule-based baseline.\n"
            "3. Рекомендуемое действие: персональный контакт.\n"
            "4. Почему это действие: несколько негативных сигналов.\n"
            "5. Ограничения / что проверить перед запуском: это предварительная оценка."
        )


def make_state(intent: str, **overrides: object) -> AgentState:
    values = {
        "request_id": "request-1",
        "message_text": "test message",
        "intent": intent,
        "intent_confidence": 0.9,
        "router_source": "rules",
    }
    values.update(overrides)
    return AgentState(**values)


@pytest.mark.asyncio
async def test_supervisor_routes_rag_question_to_rag_agent() -> None:
    rag_agent = FakeAgent("rag_agent")
    supervisor = SupervisorAgent(rag_agent=rag_agent)

    state = await supervisor.run(make_state("rag_question"))

    assert state.selected_agent == "rag_agent"
    assert rag_agent.calls == 1
    assert state.trace[0]["agent"] == "supervisor"
    assert state.trace[0]["action"] == "handoff"


@pytest.mark.asyncio
async def test_supervisor_routes_summarization_to_summarization_agent() -> None:
    summarization_agent = FakeAgent("summarization_agent")
    supervisor = SupervisorAgent(summarization_agent=summarization_agent)

    state = await supervisor.run(make_state("summarization"))

    assert state.selected_agent == "summarization_agent"
    assert summarization_agent.calls == 1


@pytest.mark.asyncio
async def test_supervisor_routes_customer_analysis_to_customer_analysis_agent() -> None:
    customer_agent = FakeAgent("customer_analysis_agent")
    supervisor = SupervisorAgent(customer_analysis_agent=customer_agent)

    state = await supervisor.run(make_state("customer_analysis", customer_id="123"))

    assert state.selected_agent == "customer_analysis_agent"
    assert customer_agent.calls == 1


@pytest.mark.asyncio
async def test_supervisor_routes_unsupported_to_fallback() -> None:
    supervisor = SupervisorAgent()

    state = await supervisor.run(make_state("unsupported"))

    assert state.selected_agent == "fallback"
    assert "не могу надежно обработать" in (state.answer or "")


@pytest.mark.asyncio
async def test_customer_analysis_agent_with_customer_id_returns_tool_results() -> None:
    agent = CustomerAnalysisAgent(llm_service=FakeLLMService())  # type: ignore[arg-type]

    state = await agent.run(make_state("customer_analysis", customer_id="123"))

    assert state.selected_agent == "customer_analysis_agent"
    assert state.tool_results["customer_profile"]["customer_id"] == "123"
    assert state.tool_results["churn_score"]["risk_level"] == "high"
    assert state.tool_results["next_best_action"]["action"] == (
        "reactivation_offer_or_personal_contact"
    )
    assert "Краткий статус клиента" in (state.answer or "")


@pytest.mark.asyncio
async def test_customer_analysis_agent_missing_customer_id_returns_controlled_answer() -> None:
    agent = CustomerAnalysisAgent(llm_service=FakeLLMService())  # type: ignore[arg-type]

    state = await agent.run(make_state("customer_analysis"))

    assert state.tool_results["customer_profile"]["error"] == "missing_customer_id"
    assert "нужен customer_id" in (state.answer or "")


@pytest.mark.asyncio
async def test_guardrails_add_missing_sources_for_rag_question() -> None:
    state = await validate_agent_result(make_state("rag_question", answer="answer", sources=[]))

    assert "missing_sources" in state.validation_errors


@pytest.mark.asyncio
async def test_guardrails_add_missing_customer_id() -> None:
    state = await validate_agent_result(
        make_state("customer_analysis", answer="answer", tool_results={})
    )

    assert "missing_customer_id" in state.validation_errors
    assert "missing_customer_tools" in state.validation_errors


@pytest.mark.asyncio
async def test_guardrails_mark_high_risk_customer_for_human_review() -> None:
    agent = CustomerAnalysisAgent(llm_service=FakeLLMService())  # type: ignore[arg-type]
    state = await agent.run(make_state("customer_analysis", customer_id="123"))

    state = await validate_agent_result(state)

    assert state.needs_human_review is True
    assert state.validation_errors == []

