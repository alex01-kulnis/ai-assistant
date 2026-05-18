from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from app.core.config import get_settings
from app.integrations.telegram.formatters import format_answer, format_sources
from app.integrations.telegram.service import (
    TelegramBackendUnavailableError,
    TelegramChatClient,
    TelegramLLMUnavailableError,
    TelegramSessionStore,
    is_user_allowed,
    parse_allowed_user_ids,
)

logger = logging.getLogger(__name__)

NO_ACCESS_MESSAGE = "У вас нет доступа к этому боту."
BACKEND_UNAVAILABLE_MESSAGE = "Backend сейчас недоступен. Проверь, что FastAPI запущен."
LLM_UNAVAILABLE_MESSAGE = "LLM сейчас недоступна. Проверь Ollama."
UNKNOWN_ERROR_MESSAGE = "Произошла ошибка при обработке сообщения."


def create_router(
    *,
    chat_client: TelegramChatClient | None = None,
    session_store: TelegramSessionStore | None = None,
) -> Router:
    settings = get_settings()
    allowed_user_ids = parse_allowed_user_ids(settings.TELEGRAM_ALLOWED_USER_IDS)
    chat_client = chat_client or TelegramChatClient()
    session_store = session_store or TelegramSessionStore()
    router = Router()

    @router.message(Command("start"))
    async def start(message: Message) -> None:
        if not _has_access(message, allowed_user_ids):
            await message.answer(NO_ACCESS_MESSAGE)
            return

        await message.answer(
            "Привет. Я SupportOps AI Agent bot.\n\n"
            "Я отвечаю на вопросы по загруженной knowledge base. "
            "Задайте вопрос обычным сообщением."
        )

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        if not _has_access(message, allowed_user_ids):
            await message.answer(NO_ACCESS_MESSAGE)
            return

        await message.answer(
            "Команды:\n"
            "/start - описание бота\n"
            "/help - помощь\n"
            "/new - начать новый диалог\n"
            "/sources - источники последнего ответа\n\n"
            "Примеры вопросов:\n"
            "Как оформить возврат?\n"
            "Что делать, если оплата не проходит?\n"
            "Где мой заказ 12345?"
        )

    @router.message(Command("new"))
    async def new_conversation(message: Message) -> None:
        if not _has_access(message, allowed_user_ids):
            await message.answer(NO_ACCESS_MESSAGE)
            return

        telegram_user_id = _telegram_user_id(message)
        if telegram_user_id is not None:
            session_store.reset_conversation(telegram_user_id)
        await message.answer("Новый диалог начат.")

    @router.message(Command("sources"))
    async def sources(message: Message) -> None:
        if not _has_access(message, allowed_user_ids):
            await message.answer(NO_ACCESS_MESSAGE)
            return

        telegram_user_id = _telegram_user_id(message)
        last_sources = (
            []
            if telegram_user_id is None
            else session_store.get_last_sources(telegram_user_id)
        )
        await message.answer(format_sources(last_sources))

    @router.message(F.text)
    async def text_message(message: Message) -> None:
        telegram_user_id = _telegram_user_id(message)
        logger.info("Incoming Telegram message from user_id=%s", telegram_user_id)

        if not _has_access(message, allowed_user_ids):
            await message.answer(NO_ACCESS_MESSAGE)
            return
        if telegram_user_id is None or message.text is None:
            await message.answer(UNKNOWN_ERROR_MESSAGE)
            return

        await message.bot.send_chat_action(
            chat_id=message.chat.id,
            action=ChatAction.TYPING,
        )

        try:
            chat_response = await chat_client.ask(
                message=message.text,
                conversation_id=session_store.get_conversation_id(telegram_user_id),
            )
        except TelegramBackendUnavailableError:
            logger.exception("Telegram backend request failed")
            await message.answer(BACKEND_UNAVAILABLE_MESSAGE)
            return
        except TelegramLLMUnavailableError:
            logger.exception("Telegram LLM request failed")
            await message.answer(LLM_UNAVAILABLE_MESSAGE)
            return
        except Exception:
            logger.exception("Unexpected Telegram message handling error")
            await message.answer(UNKNOWN_ERROR_MESSAGE)
            return

        session_store.set_conversation_id(telegram_user_id, chat_response.conversation_id)
        session_store.set_last_sources(telegram_user_id, chat_response.sources)
        await message.answer(format_answer(chat_response.answer, chat_response.sources))

    return router


def _has_access(message: Message, allowed_user_ids: set[int] | None) -> bool:
    telegram_user_id = _telegram_user_id(message)
    if telegram_user_id is None:
        return False
    return is_user_allowed(telegram_user_id, allowed_user_ids)


def _telegram_user_id(message: Message) -> int | None:
    if message.from_user is None:
        return None
    return message.from_user.id
