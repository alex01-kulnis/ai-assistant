from __future__ import annotations

import logging

from opentelemetry import trace

from app.agents.state import AgentState
from app.core.tracing import set_span_attributes

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


async def validate_agent_result(state: AgentState) -> AgentState:
    with tracer.start_as_current_span("guardrails.validate") as span:
        if state.intent == "rag_question":
            if not state.answer:
                state.validation_errors.append("empty_answer")
            if not state.sources:
                state.validation_errors.append("missing_sources")

        if state.intent == "customer_analysis":
            if state.customer_id is None:
                state.validation_errors.append("missing_customer_id")
            if not state.tool_results:
                state.validation_errors.append("missing_customer_tools")
            churn_score = state.tool_results.get("churn_score", {})
            next_best_action = state.tool_results.get("next_best_action", {})
            if (
                churn_score.get("risk_level") == "high"
                and next_best_action.get("action")
                == "reactivation_offer_or_personal_contact"
            ):
                state.needs_human_review = True

        if state.intent == "summarization" and not state.answer:
            state.validation_errors.append("empty_summary")

        if state.intent == "unsupported":
            destructive_actions = {
                "delete_all",
                "drop_table",
                "truncate",
                "destructive_action",
            }
            if destructive_actions & set(state.tool_results):
                state.validation_errors.append("unsupported_destructive_tool_result")

        state.add_trace_step(
            "guardrails_agent",
            "validated",
            {
                "validation_errors": list(state.validation_errors),
                "needs_human_review": state.needs_human_review,
            },
        )
        set_span_attributes(
            span,
            {
                "agent.name": "guardrails_agent",
                "agent.intent": state.intent,
                "agent.needs_human_review": state.needs_human_review,
                "agent.validation_errors_count": len(state.validation_errors),
            },
        )
        logger.info(
            "guardrails_validated",
            extra={
                "request_id": state.request_id,
                "conversation_id": state.conversation_id,
                "message_id": state.message_id,
                "intent": state.intent,
                "selected_agent": state.selected_agent,
                "validation_errors": state.validation_errors,
                "needs_human_review": state.needs_human_review,
            },
        )
        return state

