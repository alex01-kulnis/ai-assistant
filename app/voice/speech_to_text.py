from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.voice.schemas import VoiceTranscriptionResult

logger = logging.getLogger(__name__)


class SpeechToTextError(RuntimeError):
    pass


class LocalSpeechToTextService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise SpeechToTextError(
                    "faster-whisper is not installed. Install project dependencies first."
                ) from exc
            self._model = WhisperModel(
                self.settings.VOICE_STT_MODEL,
                device=self.settings.VOICE_STT_DEVICE,
                compute_type=self.settings.VOICE_STT_COMPUTE_TYPE,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> VoiceTranscriptionResult:
        started_at = time.perf_counter()
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                language=language or self.settings.VOICE_DEFAULT_LANGUAGE,
                beam_size=5,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            logger.exception("Speech-to-text transcription failed")
            raise SpeechToTextError("Speech-to-text transcription failed.") from exc

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return VoiceTranscriptionResult(
            text=text,
            language=getattr(info, "language", language or self.settings.VOICE_DEFAULT_LANGUAGE),
            duration_seconds=getattr(info, "duration", None),
            latency_ms=latency_ms,
            stt_provider=self.settings.VOICE_STT_PROVIDER,
            stt_model=self.settings.VOICE_STT_MODEL,
        )

