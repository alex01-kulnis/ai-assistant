from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".ogg",
    ".oga",
    ".opus",
    ".webm",
    ".m4a",
    ".mp3",
    ".mp4",
}


class AudioConversionError(RuntimeError):
    pass


class AudioConverter:
    def __init__(self, output_dir: Path | None = None) -> None:
        settings = get_settings()
        self.output_dir = output_dir or Path(settings.VOICE_AUDIO_TMP_DIR)

    def convert_to_wav(self, input_path: Path) -> Path:
        suffix = input_path.suffix.casefold()
        if suffix == ".wav":
            return input_path
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            raise AudioConversionError(f"Unsupported audio extension: {suffix or '<none>'}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{input_path.stem}-{uuid.uuid4().hex}.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AudioConversionError(
                "ffmpeg is not installed. Install ffmpeg to process Telegram voice/audio."
            ) from exc

        if completed.returncode != 0:
            logger.error(
                "ffmpeg audio conversion failed",
                extra={
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "stderr": completed.stderr[-1000:],
                },
            )
            raise AudioConversionError("Audio conversion failed.")

        return output_path

