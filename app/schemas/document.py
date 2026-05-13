from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    status: str


class DocumentListItem(BaseModel):
    id: str
    filename: str
    content_type: str
    status: str
    chunks_count: int
    created_at: datetime
