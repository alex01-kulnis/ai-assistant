from __future__ import annotations

import logging

from aiogram import Bot, Dispatcher

from app.core.config import Settings, get_settings
from app.integrations.telegram.handlers import create_router

logger = logging.getLogger(__name__)


async def run_polling(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    bot = Bot(token=settings.TELEGRAM_BOT_TOKEN)
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())

    logger.info("Starting Telegram bot polling")
    await dispatcher.start_polling(bot)
