from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.chat import get_llm_service, get_support_agent
from app.api.routes.voice import get_voice_service
from app.core.config import Settings, get_settings
from app.integrations.telegram.router import get_telegram_client
from app.main import app
from app.schemas.chat import ChatResponse
from app.voice.schemas import VoiceChatResponse


class FakeTelegramClient:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[int, str]] = []
        self.download_calls: list[str] = []

    async def get_file(self, file_id: str) -> Any:
        return type("TelegramFile", (), {"file_path": "voice/file_1.ogg"})()

    async def download_file(self, file_path: str, destination: Path) -> Path:
        self.download_calls.append(file_path)
        destination.mkdir(parents=True, exist_ok=True)
        audio_path = destination / "voice.ogg"
        audio_path.write_bytes(b"audio")
        return audio_path

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent_messages.append((chat_id, text))


class FakeVoiceService:
    def __init__(self, response: VoiceChatResponse | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response or VoiceChatResponse(
            transcript="Клиент просит возврат",
            status="success",
            answer="Voice answer",
            agent_run_id="run-1",
            stt_provider="local_whisper",
            stt_model="base",
            stt_latency_ms=100,
        )

    async def handle_voice_message(self, **kwargs: Any) -> VoiceChatResponse:
        self.calls.append(kwargs)
        audio_path = kwargs.get("audio_path")
        cleanup = kwargs.get("cleanup", True)
        if cleanup and isinstance(audio_path, Path) and audio_path.exists():
            audio_path.unlink()
        return self.response


class FakeSupportAgent:
    async def chat(self, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            conversation_id="conversation-1",
            message_id="message-1",
            answer="Text answer",
            sources=[],
            agent_run_id="run-1",
        )


class FakeLLMService:
    model_name = "fake-model"

    async def generate_chat_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
    ) -> str:
        return "LLM answer"


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def enable_telegram(fake_client: FakeTelegramClient, voice_service: FakeVoiceService) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        TELEGRAM_ENABLED=True,
        TELEGRAM_BOT_TOKEN="test-token",
        TELEGRAM_WEBHOOK_SECRET="secret",
    )
    app.dependency_overrides[get_telegram_client] = lambda: fake_client
    app.dependency_overrides[get_voice_service] = lambda: voice_service
    app.dependency_overrides[get_support_agent] = lambda: FakeSupportAgent()
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()


def test_telegram_webhook_text_message_calls_chat_and_sends_answer() -> None:
    fake_client = FakeTelegramClient()
    enable_telegram(fake_client, FakeVoiceService())
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        json={
            "update_id": 1,
            "message": {"chat": {"id": 123}, "text": "Как оформить возврат?"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.sent_messages == [(123, "Text answer")]


def test_telegram_webhook_voice_message_uses_voice_service() -> None:
    fake_client = FakeTelegramClient()
    fake_voice_service = FakeVoiceService()
    enable_telegram(fake_client, fake_voice_service)
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        json={
            "update_id": 1,
            "message": {
                "chat": {"id": 123},
                "voice": {"file_id": "file-1", "duration": 1, "mime_type": "audio/ogg"},
            },
        },
    )

    assert response.status_code == 200
    assert fake_client.download_calls == ["voice/file_1.ogg"]
    assert fake_voice_service.calls
    assert fake_client.sent_messages == [(123, "Voice answer")]


def test_telegram_voice_needs_human_review_sends_safe_message() -> None:
    fake_client = FakeTelegramClient()
    fake_voice_service = FakeVoiceService(
        VoiceChatResponse(
            transcript="refund",
            status="needs_human_review",
            answer="Unsafe direct answer",
            review_reason="high_risk_action",
            stt_provider="local_whisper",
            stt_model="base",
            stt_latency_ms=100,
        )
    )
    enable_telegram(fake_client, fake_voice_service)
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        json={
            "message": {
                "chat": {"id": 123},
                "voice": {"file_id": "file-1", "mime_type": "audio/ogg"},
            }
        },
    )

    assert response.status_code == 200
    assert fake_client.sent_messages == [
        (123, "Запрос требует проверки специалистом. Я передал его на review.")
    ]


def test_telegram_voice_failed_transcription_sends_fallback() -> None:
    fake_client = FakeTelegramClient()
    fake_voice_service = FakeVoiceService(
        VoiceChatResponse(
            transcript="",
            status="transcription_failed",
            answer=None,
            stt_provider="local_whisper",
            stt_model="base",
            stt_latency_ms=100,
        )
    )
    enable_telegram(fake_client, fake_voice_service)
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        json={
            "message": {
                "chat": {"id": 123},
                "voice": {"file_id": "file-1", "mime_type": "audio/ogg"},
            }
        },
    )

    assert response.status_code == 200
    assert "Не удалось распознать" in fake_client.sent_messages[0][1]


def test_telegram_unknown_update_returns_ok() -> None:
    fake_client = FakeTelegramClient()
    enable_telegram(fake_client, FakeVoiceService())
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
        json={"update_id": 1},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert fake_client.sent_messages == []


def test_telegram_webhook_rejects_invalid_secret() -> None:
    fake_client = FakeTelegramClient()
    enable_telegram(fake_client, FakeVoiceService())
    client = TestClient(app)

    response = client.post(
        "/api/v1/telegram/webhook",
        headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        json={"update_id": 1},
    )

    assert response.status_code == 403


def test_telegram_webhook_returns_503_when_disabled() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        TELEGRAM_ENABLED=False,
        TELEGRAM_BOT_TOKEN="test-token",
    )
    app.dependency_overrides[get_telegram_client] = lambda: FakeTelegramClient()
    app.dependency_overrides[get_voice_service] = lambda: FakeVoiceService()
    app.dependency_overrides[get_support_agent] = lambda: FakeSupportAgent()
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)

    response = client.post("/api/v1/telegram/webhook", json={"update_id": 1})

    assert response.status_code == 503
