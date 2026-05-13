from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class VectorChunkInput(BaseModel):
    document_id: uuid.UUID | str
    chunk_id: uuid.UUID | str
    chunk_index: int = Field(ge=0)
    filename: str
    page_number: int | None = Field(default=None, ge=1)
    text: str
    vector: list[float] = Field(min_length=384, max_length=384)


class RetrievedChunk(BaseModel):
    point_id: str
    score: float
    text: str
    document_id: str
    chunk_id: str
    filename: str
    page_number: int | None
    chunk_index: int
