from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.integrations.telegram.client import TelegramClient
from app.integrations.telegram.schemas import TelegramUpdate
from app.schemas.chat import ChatRequest
from app.services.chat_workflow_service import ChatWorkflowService
from app.voice.service import VoiceService


@dataclass(frozen=True)
class ChatClientResponse:
    conversation_id: str
    message_id: str
    answer: str
    sources: list[dict[str, Any]]


class TelegramChatClientError(RuntimeError):
    pass


class TelegramBackendUnavailableError(TelegramChatClientError):
    pass


class TelegramLLMUnavailableError(TelegramChatClientError):
    pass


class TelegramInvalidResponseError(TelegramChatClientError):
    pass


@dataclass
class TelegramSessionStore:
    conversation_ids: dict[int, str] = field(default_factory=dict)
    last_sources: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    def get_conversation_id(self, telegram_user_id: int) -> str | None:
        return self.conversation_ids.get(telegram_user_id)

    def set_conversation_id(self, telegram_user_id: int, conversation_id: str) -> None:
        self.conversation_ids[telegram_user_id] = conversation_id

    def reset_conversation(self, telegram_user_id: int) -> None:
        self.conversation_ids.pop(telegram_user_id, None)
        self.last_sources.pop(telegram_user_id, None)

    def get_last_sources(self, telegram_user_id: int) -> list[dict[str, Any]]:
        return self.last_sources.get(telegram_user_id, [])

    def set_last_sources(self, telegram_user_id: int, sources: list[dict[str, Any]]) -> None:
        self.last_sources[telegram_user_id] = sources


class TelegramChatClient:
    def __init__(
        self,
        *,
        use_backend_http: bool | None = None,
        backend_chat_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.use_backend_http = (
            settings.TELEGRAM_USE_BACKEND_HTTP
            if use_backend_http is None
            else use_backend_http
        )
        self.backend_chat_url = backend_chat_url or settings.TELEGRAM_BACKEND_CHAT_URL
        self._http_client = http_client
        self.timeout = timeout

    async def ask(self, message: str, conversation_id: str | None) -> ChatClientResponse:
        if self.use_backend_http:
            return await self._ask_via_http(message=message, conversation_id=conversation_id)

        return await self._ask_direct(message=message, conversation_id=conversation_id)

    async def _ask_via_http(
        self,
        *,
        message: str,
        conversation_id: str | None,
    ) -> ChatClientResponse:
        payload = {
            "conversation_id": conversation_id,
            "message": message,
        }

        try:
            if self._http_client is not None:
                response = await self._http_client.post(
                    self.backend_chat_url,
                    json=payload,
                    timeout=self.timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.backend_chat_url, json=payload)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
            raise TelegramBackendUnavailableError("Backend is unavailable.") from exc

        if response.status_code == 502:
            raise TelegramLLMUnavailableError(_extract_error_detail(response))
        if response.status_code >= 400:
            raise TelegramBackendUnavailableError(_extract_error_detail(response))

        try:
            response_data = response.json()
        except ValueError as exc:
            raise TelegramInvalidResponseError("Backend returned invalid JSON.") from exc

        return _parse_chat_client_response(response_data)

    async def _ask_direct(
        self,
        *,
        message: str,
        conversation_id: str | None,
    ) -> ChatClientResponse:
        from app.agents.support_agent import SupportAgent, SupportAgentError
        from app.db.session import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                response = await SupportAgent().chat(
                    conversation_id=conversation_id,
                    message=message,
                    session=session,
                )
        except SupportAgentError as exc:
            if exc.status_code == 502:
                raise TelegramLLMUnavailableError(exc.message) from exc
            raise TelegramChatClientError(exc.message) from exc

        return ChatClientResponse(
            conversation_id=response.conversation_id,
            message_id=response.message_id,
            answer=response.answer,
            sources=[source.model_dump() for source in response.sources],
        )


def parse_allowed_user_ids(raw_user_ids: str | None) -> set[int] | None:
    if raw_user_ids is None or not raw_user_ids.strip():
        return None

    return {
        int(raw_user_id.strip())
        for raw_user_id in raw_user_ids.split(",")
        if raw_user_id.strip()
    }


def is_user_allowed(telegram_user_id: int, allowed_user_ids: set[int] | None) -> bool:
    if allowed_user_ids is None:
        return True
    return telegram_user_id in allowed_user_ids


def _parse_chat_client_response(response_data: Any) -> ChatClientResponse:
    if not isinstance(response_data, dict):
        raise TelegramInvalidResponseError("Backend returned unexpected response format.")

    try:
        return ChatClientResponse(
            conversation_id=str(response_data["conversation_id"]),
            message_id=str(response_data["message_id"]),
            answer=str(response_data["answer"]),
            sources=list(response_data.get("sources", [])),
        )
    except KeyError as exc:
        raise TelegramInvalidResponseError("Backend response misses required fields.") from exc


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        response_data = response.json()
    except ValueError:
        return response.text

    if isinstance(response_data, dict):
        detail = response_data.get("detail")
        if detail:
            return str(detail)

    return response.text


SAFE_REVIEW_MESSAGE = "Запрос требует проверки специалистом. Я передал его на review."
TRANSCRIPTION_FAILED_MESSAGE = (
    "Не удалось распознать голосовое сообщение. "
    "Попробуйте записать его ещё раз или отправьте текстом."
)


class TelegramWebhookService:
    def __init__(
        self,
        *,
        telegram_client: TelegramClient,
        voice_service: VoiceService,
        chat_workflow: ChatWorkflowService,
        audio_tmp_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.telegram_client = telegram_client
        self.voice_service = voice_service
        self.chat_workflow = chat_workflow
        self.audio_tmp_dir = audio_tmp_dir or Path(settings.VOICE_AUDIO_TMP_DIR)

    async def handle_update(
        self,
        *,
        update: TelegramUpdate,
        session: AsyncSession,
    ) -> None:
        message = update.message
        if message is None:
            return

        chat_id = message.chat.id
        if message.text:
            await self._handle_text_message(
                chat_id=chat_id,
                text=message.text,
                session=session,
            )
            return

        if message.voice:
            await self._handle_voice_message(
                chat_id=chat_id,
                file_id=message.voice.file_id,
                session=session,
            )

    async def _handle_text_message(
        self,
        *,
        chat_id: int,
        text: str,
        session: AsyncSession,
    ) -> None:
        response = await self.chat_workflow.process(
            request=ChatRequest(message=text, user_id=str(chat_id)),
            session=session,
        )
        await self.telegram_client.send_message(
            chat_id=chat_id,
            text=SAFE_REVIEW_MESSAGE if response.needs_human_review else response.answer,
        )

    async def _handle_voice_message(
        self,
        *,
        chat_id: int,
        file_id: str,
        session: AsyncSession,
    ) -> None:
        telegram_file = await self.telegram_client.get_file(file_id)
        audio_path = await self.telegram_client.download_file(
            telegram_file.file_path,
            self.audio_tmp_dir,
        )
        voice_response = await self.voice_service.handle_voice_message(
            audio_path=audio_path,
            session=session,
            chat_workflow=self.chat_workflow,
            user_id=str(chat_id),
            cleanup=True,
        )
        if voice_response.status == "needs_human_review":
            await self.telegram_client.send_message(chat_id=chat_id, text=SAFE_REVIEW_MESSAGE)
            return
        if voice_response.status == "transcription_failed":
            await self.telegram_client.send_message(
                chat_id=chat_id,
                text=TRANSCRIPTION_FAILED_MESSAGE,
            )
            return
        await self.telegram_client.send_message(
            chat_id=chat_id,
            text=voice_response.answer or TRANSCRIPTION_FAILED_MESSAGE,
        )
