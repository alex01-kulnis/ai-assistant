from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings


class OllamaLLMError(RuntimeError):
    pass


class OllamaTimeoutError(OllamaLLMError):
    pass


class OllamaConnectionError(OllamaLLMError):
    pass


class OllamaInvalidResponseError(OllamaLLMError):
    pass


class OllamaLLMService:
    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.timeout = timeout
        self._client = client

    async def generate_chat_response(
        self,
        messages: list[dict],
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response_data = await self._post_json("/api/chat", payload)
        message = response_data.get("message")
        if not isinstance(message, dict):
            raise OllamaInvalidResponseError("Ollama response does not contain message object.")

        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaInvalidResponseError("Ollama response does not contain message content.")

        return content

    async def health_check(self) -> bool:
        try:
            response = await self._get("/api/tags")
            return 200 <= response.status_code < 300
        except OllamaLLMError:
            return False

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._post(path, payload)
        if response.status_code >= 400:
            raise OllamaInvalidResponseError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            )

        try:
            response_data = response.json()
        except ValueError as exc:
            raise OllamaInvalidResponseError("Ollama returned invalid JSON response.") from exc

        if not isinstance(response_data, dict):
            raise OllamaInvalidResponseError("Ollama returned unexpected response format.")

        return response_data

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            if self._client is not None:
                return await self._client.post(self._url(path), json=payload, timeout=self.timeout)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.post(self._url(path), json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama request timed out.") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError("Could not connect to Ollama.") from exc
        except httpx.NetworkError as exc:
            raise OllamaConnectionError("Ollama network request failed.") from exc

    async def _get(self, path: str) -> httpx.Response:
        try:
            if self._client is not None:
                return await self._client.get(self._url(path), timeout=self.timeout)

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                return await client.get(self._url(path))
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Ollama request timed out.") from exc
        except httpx.ConnectError as exc:
            raise OllamaConnectionError("Could not connect to Ollama.") from exc
        except httpx.NetworkError as exc:
            raise OllamaConnectionError("Ollama network request failed.") from exc

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"
