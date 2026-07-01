from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.support_agent import SupportAgent
from app.api.routes.chat import get_llm_service, get_support_agent
from app.api.routes.voice import get_voice_service
from app.core.config import Settings, get_settings
from app.core.tracing import set_span_attributes
from app.db.session import get_db_session
from app.integrations.telegram.client import TelegramClient
from app.integrations.telegram.schemas import TelegramUpdate
from app.integrations.telegram.service import TelegramWebhookService
from app.services.chat_workflow_service import ChatWorkflowService
from app.services.llm_service import OllamaLLMService
from app.voice.service import VoiceService

router = APIRouter(prefix="/api/v1/telegram", tags=["telegram"])
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def get_telegram_client() -> TelegramClient:
    return TelegramClient()


@router.post("/webhook")
async def telegram_webhook(
    update: TelegramUpdate,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    support_agent: Annotated[SupportAgent, Depends(get_support_agent)],
    llm_service: Annotated[OllamaLLMService, Depends(get_llm_service)],
    voice_service: Annotated[VoiceService, Depends(get_voice_service)],
    telegram_client: Annotated[TelegramClient, Depends(get_telegram_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_telegram_bot_api_secret_token: Annotated[
        str | None,
        Header(alias="X-Telegram-Bot-Api-Secret-Token"),
    ] = None,
) -> dict[str, bool]:
    if not settings.TELEGRAM_ENABLED:
        raise HTTPException(status_code=503, detail="Telegram webhook is disabled.")
    if (
        settings.TELEGRAM_WEBHOOK_SECRET
        and x_telegram_bot_api_secret_token != settings.TELEGRAM_WEBHOOK_SECRET
    ):
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret.")

    with tracer.start_as_current_span("telegram.webhook") as span:
        message_type = "unknown"
        if update.message and update.message.text:
            message_type = "text"
        elif update.message and update.message.voice:
            message_type = "voice"
        set_span_attributes(
            span,
            {
                "telegram.update_id": update.update_id,
                "telegram.message_type": message_type,
            },
        )
        logger.info(
            "Telegram update received",
            extra={"update_id": update.update_id, "message_type": message_type},
        )
        service = TelegramWebhookService(
            telegram_client=telegram_client,
            voice_service=voice_service,
            chat_workflow=ChatWorkflowService(
                support_agent=support_agent,
                llm_service=llm_service,
            ),
        )
        await service.handle_update(update=update, session=session)
    return {"ok": True}

