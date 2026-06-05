from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentReindexResponse,
    DocumentUploadResponse,
)
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


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[DocumentIngestionService, Depends(get_document_ingestion_service)],
) -> DocumentDeleteResponse:
    try:
        return await service.delete_document(
            document_id=document_id,
            session=session,
        )
    except DocumentIndexingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc


@router.post("/{document_id}/reindex", response_model=DocumentReindexResponse)
async def reindex_document(
    document_id: str,
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    service: Annotated[DocumentIngestionService, Depends(get_document_ingestion_service)],
) -> DocumentReindexResponse:
    try:
        content = await file.read()
        return await service.reindex_document(
            document_id=document_id,
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
            session=session,
        )
    except DocumentIndexingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
