from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import SupportAgent, SupportAgentError
from app.core.tracing import set_span_attributes
from app.db.session import get_db_session
from app.routing.classifier import classify_intent
from app.routing.schemas import IntentResult
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import OllamaLLMError, OllamaLLMService
from app.services.tools.customer_tools import get_mock_customer_profile

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

UNSUPPORTED_REQUEST_MESSAGE = (
    "Я не могу надежно обработать этот запрос в текущем сценарии. "
    "Попробуйте задать вопрос по документам, тикету или клиентскому профилю."
)


def get_support_agent() -> SupportAgent:
    return SupportAgent()


def get_llm_service() -> OllamaLLMService:
    return OllamaLLMService()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    agent: Annotated[SupportAgent, Depends(get_support_agent)],
    llm_service: Annotated[OllamaLLMService, Depends(get_llm_service)],
) -> ChatResponse:
    intent_result = await classify_intent(request, llm_service=llm_service)
    _log_intent_result(request, intent_result)

    with tracer.start_as_current_span("chat.route") as span:
        set_span_attributes(
            span,
            {
                "intent": intent_result.intent,
                "intent_confidence": intent_result.confidence,
                "router_source": intent_result.source,
                "message_length": len(request.message_text),
            },
        )

        if intent_result.intent == "rag_question":
            return await _run_rag_chat(
                request=request,
                session=session,
                agent=agent,
                intent_result=intent_result,
            )

        if intent_result.intent == "summarization":
            return await _run_summarization(
                request=request,
                llm_service=llm_service,
                intent_result=intent_result,
            )

        if intent_result.intent == "customer_analysis":
            return _run_customer_analysis(request=request, intent_result=intent_result)

        return _build_chat_response(
            answer=UNSUPPORTED_REQUEST_MESSAGE,
            conversation_id=request.conversation_id,
            intent_result=intent_result,
        )


async def _run_rag_chat(
    *,
    request: ChatRequest,
    session: AsyncSession,
    agent: SupportAgent,
    intent_result: IntentResult,
) -> ChatResponse:
    try:
        response = await agent.chat(
            conversation_id=request.conversation_id,
            message=request.message_text,
            session=session,
        )
        return _with_intent_metadata(response, intent_result)
    except SupportAgentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


async def _run_summarization(
    *,
    request: ChatRequest,
    llm_service: OllamaLLMService,
    intent_result: IntentResult,
) -> ChatResponse:
    if not request.selected_text:
        return _build_chat_response(
            answer=(
                "Для суммаризации передайте текст в поле selected_text. "
                "Сейчас нечего кратко пересказать."
            ),
            conversation_id=request.conversation_id,
            intent_result=intent_result,
        )

    messages = [
        {
            "role": "system",
            "content": (
                "Ты support assistant. Кратко суммаризируй только переданный текст. "
                "Не добавляй факты извне и отвечай на русском языке."
            ),
        },
        {
            "role": "user",
            "content": f"TEXT:\n{request.selected_text}\n\nSUMMARY:",
        },
    ]
    try:
        answer = await llm_service.generate_chat_response(messages, temperature=0.1)
    except OllamaLLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    return _build_chat_response(
        answer=answer.strip(),
        conversation_id=request.conversation_id,
        intent_result=intent_result,
    )


def _run_customer_analysis(
    *,
    request: ChatRequest,
    intent_result: IntentResult,
) -> ChatResponse:
    profile = get_mock_customer_profile(request.customer_id)
    if request.customer_id is None:
        answer = (
            "Для точного анализа клиента нужен customer_id. "
            "В текущем MVP можно оценить профиль клиента, риск оттока и следующее действие, "
            "если передать customer_id."
        )
    else:
        answer = (
            f"Клиент {request.customer_id}: сегмент - {profile['segment']}, "
            f"риск оттока - {profile['churn_risk']}. "
            f"Рекомендация: {profile['recommended_action']}"
        )

    return _build_chat_response(
        answer=answer,
        conversation_id=request.conversation_id,
        intent_result=intent_result,
    )


def _with_intent_metadata(
    response: ChatResponse,
    intent_result: IntentResult,
) -> ChatResponse:
    return response.model_copy(
        update={
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
            "router_source": intent_result.source,
        }
    )


def _build_chat_response(
    *,
    answer: str,
    conversation_id: str | None,
    intent_result: IntentResult,
) -> ChatResponse:
    return ChatResponse(
        conversation_id=conversation_id or str(uuid.uuid4()),
        message_id=str(uuid.uuid4()),
        answer=answer,
        sources=[],
        intent=intent_result.intent,
        intent_confidence=intent_result.confidence,
        router_source=intent_result.source,
    )


def _log_intent_result(request: ChatRequest, intent_result: IntentResult) -> None:
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
        },
    )
