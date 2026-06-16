from __future__ import annotations

import logging

from opentelemetry import trace

from app.agents.base import BaseAgent
from app.agents.guardrails import validate_agent_result
from app.agents.state import AgentState
from app.core.tracing import set_span_attributes
from app.tools.agent_tools import build_fallback_answer

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def __init__(
        self,
        *,
        rag_agent: BaseAgent | None = None,
        summarization_agent: BaseAgent | None = None,
        customer_analysis_agent: BaseAgent | None = None,
    ) -> None:
        self._agents = {
            "rag_question": rag_agent,
            "summarization": summarization_agent,
            "customer_analysis": customer_analysis_agent,
        }

    async def run(self, state: AgentState) -> AgentState:
        with tracer.start_as_current_span("supervisor.run") as span:
            selected_agent = self._select_agent_name(state.intent)
            state.current_agent = self.name
            state.selected_agent = selected_agent
            state.add_trace_step(
                self.name,
                "handoff",
                {
                    "intent": state.intent,
                    "selected_agent": selected_agent,
                    "confidence": state.intent_confidence,
                    "router_source": state.router_source,
                },
            )
            set_span_attributes(
                span,
                {
                    "agent.name": self.name,
                    "agent.selected_agent": selected_agent,
                    "agent.intent": state.intent,
                    "agent.needs_human_review": state.needs_human_review,
                    "agent.validation_errors_count": len(state.validation_errors),
                },
            )
            logger.info(
                "supervisor_handoff",
                extra={
                    "request_id": state.request_id,
                    "conversation_id": state.conversation_id,
                    "message_id": state.message_id,
                    "intent": state.intent,
                    "intent_confidence": state.intent_confidence,
                    "router_source": state.router_source,
                    "selected_agent": selected_agent,
                },
            )

            agent = self._agents.get(state.intent)
            if selected_agent == "fallback" or agent is None:
                state = self._run_fallback(state)
            else:
                state = await agent.run(state)

            state = await validate_agent_result(state)
            set_span_attributes(
                span,
                {
                    "agent.needs_human_review": state.needs_human_review,
                    "agent.validation_errors_count": len(state.validation_errors),
                },
            )
            return state

    def _select_agent_name(self, intent: str) -> str:
        if intent == "rag_question":
            return "rag_agent"
        if intent == "summarization":
            return "summarization_agent"
        if intent == "customer_analysis":
            return "customer_analysis_agent"
        return "fallback"

    def _run_fallback(self, state: AgentState) -> AgentState:
        state.current_agent = "fallback"
        state.selected_agent = "fallback"
        state.answer = build_fallback_answer()
        state.sources = []
        state.add_trace_step("fallback", "controlled_response")
        return state

