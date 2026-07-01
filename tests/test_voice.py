from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.routes.voice import get_voice_service
from app.core.config import Settings, get_settings
from app.main import app
from app.voice.audio_converter import AudioConversionError, AudioConverter
from app.voice.schemas import VoiceChatResponse, VoiceTranscriptionResult


class FakeVoiceService:
    def __init__(self) -> None:
        self.transcribe_calls: list[Path] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.transcription = VoiceTranscriptionResult(
            text="Клиент просит возврат",
            language="ru",
            duration_seconds=2.0,
            latency_ms=123,
            stt_provider="local_whisper",
            stt_model="base",
        )
        self.chat_response = VoiceChatResponse(
            transcript="Клиент просит возврат",
            status="success",
            answer="Ответ агента",
            agent_run_id="run-1",
            stt_provider="local_whisper",
            stt_model="base",
            stt_latency_ms=123,
        )

    def transcribe_audio(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        cleanup: bool = True,
    ) -> VoiceTranscriptionResult:
        self.transcribe_calls.append(audio_path)
        if cleanup and audio_path.exists():
            audio_path.unlink()
        return self.transcription

    async def handle_voice_message(self, **kwargs: Any) -> VoiceChatResponse:
        self.chat_calls.append(kwargs)
        audio_path = kwargs.get("audio_path")
        cleanup = kwargs.get("cleanup", True)
        if cleanup and isinstance(audio_path, Path) and audio_path.exists():
            audio_path.unlink()
        return self.chat_response


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_voice_transcribe_rejects_unsupported_content_type() -> None:
    app.dependency_overrides[get_voice_service] = lambda: FakeVoiceService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("sample.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_voice_transcribe_rejects_too_large_audio() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(VOICE_MAX_AUDIO_SIZE_MB=0)
    app.dependency_overrides[get_voice_service] = lambda: FakeVoiceService()
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("sample.mp3", b"x", "audio/mpeg")},
    )

    assert response.status_code == 413


def test_voice_transcribe_calls_stt_service() -> None:
    fake_service = FakeVoiceService()
    app.dependency_overrides[get_voice_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/transcribe",
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Клиент просит возврат"
    assert fake_service.transcribe_calls


def test_voice_chat_calls_voice_service_and_returns_agent_answer() -> None:
    fake_service = FakeVoiceService()
    app.dependency_overrides[get_voice_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/chat",
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Ответ агента"
    assert fake_service.chat_calls


def test_voice_chat_passes_through_human_review_status() -> None:
    fake_service = FakeVoiceService()
    fake_service.chat_response = VoiceChatResponse(
        transcript="Клиент просит возврат",
        status="needs_human_review",
        answer="Нужна проверка",
        agent_run_id="run-1",
        review_reason="high_risk_action",
        stt_provider="local_whisper",
        stt_model="base",
        stt_latency_ms=123,
    )
    app.dependency_overrides[get_voice_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/chat",
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "needs_human_review"
    assert response.json()["review_reason"] == "high_risk_action"


def test_voice_chat_returns_controlled_status_for_empty_transcript() -> None:
    fake_service = FakeVoiceService()
    fake_service.chat_response = VoiceChatResponse(
        transcript="",
        status="transcription_failed",
        answer=None,
        stt_provider="local_whisper",
        stt_model="base",
        stt_latency_ms=123,
    )
    app.dependency_overrides[get_voice_service] = lambda: fake_service
    client = TestClient(app)

    response = client.post(
        "/api/v1/voice/chat",
        files={"file": ("sample.mp3", b"audio", "audio/mpeg")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "transcription_failed"


def test_audio_converter_skips_wav_conversion(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"wav")

    assert AudioConverter(output_dir=tmp_path).convert_to_wav(audio_path) == audio_path


def test_audio_converter_calls_ffmpeg_for_ogg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.ogg"
    audio_path.write_bytes(b"ogg")
    captured_command: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_path = AudioConverter(output_dir=tmp_path).convert_to_wav(audio_path)

    assert captured_command[0] == "ffmpeg"
    assert output_path.suffix == ".wav"


def test_audio_converter_raises_controlled_error_on_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.ogg"
    audio_path.write_bytes(b"ogg")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioConversionError, match="Audio conversion failed"):
        AudioConverter(output_dir=tmp_path).convert_to_wav(audio_path)


def test_audio_converter_raises_controlled_error_when_ffmpeg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.ogg"
    audio_path.write_bytes(b"ogg")

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(AudioConversionError, match="ffmpeg is not installed"):
        AudioConverter(output_dir=tmp_path).convert_to_wav(audio_path)
