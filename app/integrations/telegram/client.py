from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
from opentelemetry import trace

from app.core.config import get_settings
from app.core.tracing import set_span_attributes
from app.integrations.telegram.schemas import TelegramFile

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class TelegramClientError(RuntimeError):
    pass


class TelegramClient:
    def __init__(
        self,
        *,
        token: str | None = None,
        api_base_url: str | None = None,
        file_base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        settings = get_settings()
        self.token = token or settings.TELEGRAM_BOT_TOKEN
        self.api_base_url = (api_base_url or settings.TELEGRAM_API_BASE_URL).rstrip("/")
        self.file_base_url = (file_base_url or settings.TELEGRAM_FILE_BASE_URL).rstrip("/")
        self._http_client = http_client
        self.timeout = timeout

    async def get_file(self, file_id: str) -> TelegramFile:
        token = self._require_token()
        response_data = await self._get_json(
            f"{self.api_base_url}/bot{token}/getFile",
            params={"file_id": file_id},
        )
        result = response_data.get("result")
        if not isinstance(result, dict):
            raise TelegramClientError("Telegram getFile returned invalid result.")
        logger.info("Telegram getFile succeeded", extra={"file_id": file_id})
        return TelegramFile(**result)

    async def download_file(self, file_path: str, destination: Path) -> Path:
        with tracer.start_as_current_span("telegram.download_file") as span:
            token = self._require_token()
            suffix = Path(file_path).suffix or ".ogg"
            destination.mkdir(parents=True, exist_ok=True)
            output_path = destination / f"telegram-{uuid.uuid4().hex}{suffix}"
            url = f"{self.file_base_url}/bot{token}/{file_path.lstrip('/')}"
            set_span_attributes(span, {"telegram.file_path": file_path})
            try:
                if self._http_client is not None:
                    response = await self._http_client.get(url, timeout=self.timeout)
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.exception("Telegram file download failed")
                raise TelegramClientError("Telegram file download failed.") from exc

            output_path.write_bytes(response.content)
            set_span_attributes(
                span,
                {
                    "downloaded_file_size": len(response.content),
                    "destination": str(output_path),
                },
            )
            logger.info(
                "Telegram file downloaded",
                extra={"file_path": file_path, "destination": str(output_path)},
            )
            return output_path

    async def send_message(self, chat_id: int, text: str) -> None:
        with tracer.start_as_current_span("telegram.send_message") as span:
            token = self._require_token()
            set_span_attributes(span, {"telegram.chat_id": chat_id, "message_length": len(text)})
            await self._get_json(
                f"{self.api_base_url}/bot{token}/sendMessage",
                params={"chat_id": chat_id, "text": text},
            )
            logger.info("Telegram sendMessage succeeded", extra={"chat_id": chat_id})

    def _require_token(self) -> str:
        if not self.token:
            raise TelegramClientError("TELEGRAM_BOT_TOKEN is not configured.")
        return self.token

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            if self._http_client is not None:
                response = await self._http_client.get(url, params=params, timeout=self.timeout)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.exception("Telegram API request failed")
            raise TelegramClientError("Telegram API request failed.") from exc

        try:
            response_data = response.json()
        except ValueError as exc:
            raise TelegramClientError("Telegram API returned invalid JSON.") from exc

        if not isinstance(response_data, dict) or not response_data.get("ok", False):
            raise TelegramClientError("Telegram API returned an error response.")
        return response_data
