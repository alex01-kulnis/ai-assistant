from __future__ import annotations

from pydantic import BaseModel


class TelegramChat(BaseModel):
    id: int


class TelegramVoice(BaseModel):
    file_id: str
    duration: int | None = None
    mime_type: str | None = None
    file_size: int | None = None


class TelegramMessage(BaseModel):
    message_id: int | None = None
    chat: TelegramChat
    text: str | None = None
    voice: TelegramVoice | None = None


class TelegramUpdate(BaseModel):
    update_id: int | None = None
    message: TelegramMessage | None = None


class TelegramFile(BaseModel):
    file_id: str
    file_unique_id: str | None = None
    file_size: int | None = None
    file_path: str

