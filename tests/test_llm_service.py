import json

import httpx
import pytest

from app.services.llm_service import (
    OllamaConnectionError,
    OllamaInvalidResponseError,
    OllamaLLMService,
    OllamaTimeoutError,
)


@pytest.mark.asyncio
async def test_generate_chat_response_returns_message_content() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)

        assert request.url.path == "/api/chat"
        assert payload == {
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }
        return httpx.Response(200, json={"message": {"content": "Hello from Ollama"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OllamaLLMService(
            base_url="http://ollama.test",
            model_name="qwen2.5:7b",
            client=client,
        )

        response = await service.generate_chat_response(
            [{"role": "user", "content": "Hello"}],
            temperature=0.2,
        )

    assert response == "Hello from Ollama"


@pytest.mark.asyncio
async def test_generate_chat_response_rejects_invalid_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OllamaLLMService(base_url="http://ollama.test", client=client)

        with pytest.raises(
            OllamaInvalidResponseError,
            match="message content",
        ):
            await service.generate_chat_response([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_health_check_returns_true_when_ollama_is_available() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OllamaLLMService(base_url="http://ollama.test", client=client)

        is_available = await service.health_check()

    assert is_available is True


@pytest.mark.asyncio
async def test_generate_chat_response_handles_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OllamaLLMService(base_url="http://ollama.test", client=client)

        with pytest.raises(OllamaTimeoutError, match="timed out"):
            await service.generate_chat_response([{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_generate_chat_response_handles_connection_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        service = OllamaLLMService(base_url="http://ollama.test", client=client)

        with pytest.raises(OllamaConnectionError, match="connect"):
            await service.generate_chat_response([{"role": "user", "content": "Hello"}])
