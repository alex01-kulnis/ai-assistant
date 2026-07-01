from __future__ import annotations

import logging
import uuid

from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.customer_analysis_agent import CustomerAnalysisAgent
from app.agents.rag_agent import RagAgent
from app.agents.state import AgentState
from app.agents.summarization_agent import SummarizationAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.support_agent import SupportAgent
from app.core.tracing import set_span_attributes
from app.routing.classifier import classify_intent
from app.routing.schemas import IntentResult
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import OllamaLLMService

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ChatWorkflowService:
    def __init__(
        self,
        *,
        support_agent: SupportAgent,
        llm_service: OllamaLLMService,
    ) -> None:
        self.support_agent = support_agent
        self.llm_service = llm_service

    async def process(
        self,
        *,
        request: ChatRequest,
        session: AsyncSession,
    ) -> ChatResponse:
        intent_result = await classify_intent(request, llm_service=self.llm_service)
        self._log_intent_result(request, intent_result)

        state = self._build_agent_state(request=request, intent_result=intent_result)
        supervisor = SupervisorAgent(
            rag_agent=RagAgent(support_agent=self.support_agent, session=session),
            summarization_agent=SummarizationAgent(llm_service=self.llm_service),
            customer_analysis_agent=CustomerAnalysisAgent(llm_service=self.llm_service),
        )

        with tracer.start_as_current_span("chat.multi_agent_workflow") as span:
            set_span_attributes(
                span,
                {
                    "request_id": state.request_id,
                    "intent": state.intent,
                    "intent_confidence": state.intent_confidence,
                    "router_source": state.router_source,
                    "message_length": len(state.message_text),
                    "input_mode": state.input_mode,
                },
            )
            state = await supervisor.run(state)
            set_span_attributes(
                span,
                {
                    "agent.selected_agent": state.selected_agent,
                    "agent.intent": state.intent,
                    "agent.needs_human_review": state.needs_human_review,
                    "agent.validation_errors_count": len(state.validation_errors),
                },
            )
            return self._build_chat_response(state)

    def _build_agent_state(
        self,
        *,
        request: ChatRequest,
        intent_result: IntentResult,
    ) -> AgentState:
        return AgentState(
            request_id=str(uuid.uuid4()),
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            message_text=request.message_text,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            router_source=intent_result.source,
            customer_id=request.customer_id,
            ticket_id=request.ticket_id,
            document_id=request.document_id,
            action=request.action,
            selected_text=request.selected_text,
            debug=request.debug,
            input_mode=request.input_mode,
            input_audio_path=request.input_audio_path,
            input_transcript=request.input_transcript,
            stt_provider=request.stt_provider,
            stt_model=request.stt_model,
            stt_latency_ms=request.stt_latency_ms,
        )

    def _build_chat_response(self, state: AgentState) -> ChatResponse:
        return ChatResponse(
            conversation_id=state.conversation_id or str(uuid.uuid4()),
            message_id=state.message_id or str(uuid.uuid4()),
            answer=state.answer or "",
            sources=state.sources,
            agent_run_id=state.agent_run_id,
            intent=state.intent,
            intent_confidence=state.intent_confidence,
            router_source=state.router_source,
            selected_agent=state.selected_agent,
            validation_errors=state.validation_errors,
            needs_human_review=state.needs_human_review,
            trace=state.trace if state.debug else None,
        )

    def _log_intent_result(self, request: ChatRequest, intent_result: IntentResult) -> None:
        logger.info(
            "intent_classified",
            extra={
                "intent": intent_result.intent,
                "intent_confidence": intent_result.confidence,
                "router_source": intent_result.source,
                "message_length": len(request.message_text),
                "has_customer_id": request.customer_id is not None,
                "has_document_id": request.document_id is not None,
                "has_selected_text": request.selected_text is not None,
                "input_mode": request.input_mode,
            },
        )

