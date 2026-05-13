from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.document import DocumentListItem, DocumentUploadResponse
from app.services.document_ingestion_service import (
    DocumentIndexingError,
    DocumentIngestionService,
)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_document_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[DocumentIngestionService, Depends(get_document_ingestion_service)],
) -> DocumentUploadResponse:
    try:
        content = await file.read()
        return await service.index_uploaded_document(
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
            session=session,
        )
    except DocumentIndexingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.get("", response_model=list[DocumentListItem])
async def list_documents(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[DocumentIngestionService, Depends(get_document_ingestion_service)],
) -> list[DocumentListItem]:
    return await service.list_documents(session)
