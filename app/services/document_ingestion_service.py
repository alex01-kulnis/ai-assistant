from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.schemas.document import DocumentListItem, DocumentUploadResponse
from app.schemas.vector import VectorChunkInput
from app.services.chunking_service import TextChunkingService
from app.services.document_parser import DocumentParser, DocumentParsingError
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.vectorstore.qdrant_store import QdrantVectorStore, get_qdrant_vector_store

logger = logging.getLogger(__name__)


class DocumentIndexingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentIngestionService:
    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunking_service: TextChunkingService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.chunking_service = chunking_service or TextChunkingService()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_qdrant_vector_store()

    async def index_uploaded_document(
        self,
        *,
        filename: str,
        content_type: str | None,
        content: bytes,
        session: AsyncSession,
    ) -> DocumentUploadResponse:
        safe_filename = Path(filename).name
        if not safe_filename:
            raise DocumentIndexingError("Uploaded file must have a filename.", status_code=400)

        document = Document(
            id=uuid.uuid4(),
            filename=safe_filename,
            content_type=content_type or "application/octet-stream",
            status="processing",
        )
        session.add(document)
        await session.commit()

        try:
            parsed_document = self._parse_uploaded_content(
                filename=safe_filename,
                content=content,
            )
            document.content_type = parsed_document.content_type

            chunks = self.chunking_service.split_pages(parsed_document.pages)
            if not chunks:
                raise DocumentIndexingError(
                    "Document does not contain text to index.",
                    status_code=400,
                )

            embeddings = self.embedding_service.embed_documents([chunk.text for chunk in chunks])
            chunk_ids = [uuid.uuid4() for _ in chunks]
            vector_chunks = [
                VectorChunkInput(
                    document_id=document.id,
                    chunk_id=chunk_id,
                    chunk_index=chunk.chunk_index,
                    filename=document.filename,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    vector=embedding,
                )
                for chunk, chunk_id, embedding in zip(chunks, chunk_ids, embeddings, strict=True)
            ]

            qdrant_point_ids = self.vector_store.upsert_chunks(vector_chunks)

            session.add_all(
                [
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document.id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        page_number=chunk.page_number,
                        qdrant_point_id=qdrant_point_id,
                    )
                    for chunk, chunk_id, qdrant_point_id in zip(
                        chunks,
                        chunk_ids,
                        qdrant_point_ids,
                        strict=True,
                    )
                ]
            )
            document.status = "indexed"
            await session.commit()

            return DocumentUploadResponse(
                document_id=str(document.id),
                filename=document.filename,
                chunks_count=len(chunks),
                status=document.status,
            )
        except DocumentIndexingError as exc:
            await self._mark_document_failed(session=session, document_id=document.id)
            raise exc
        except (DocumentParsingError, UnicodeDecodeError) as exc:
            await self._mark_document_failed(session=session, document_id=document.id)
            raise DocumentIndexingError(str(exc), status_code=400) from exc
        except Exception as exc:
            logger.exception("Document indexing failed for document_id=%s", document.id)
            await self._mark_document_failed(session=session, document_id=document.id)
            raise DocumentIndexingError(
                "Failed to index document. Check Qdrant and embedding model availability.",
                status_code=500,
            ) from exc

    async def list_documents(self, session: AsyncSession) -> list[DocumentListItem]:
        statement = (
            select(
                Document.id,
                Document.filename,
                Document.content_type,
                Document.status,
                Document.created_at,
                func.count(DocumentChunk.id).label("chunks_count"),
            )
            .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
            .group_by(
                Document.id,
                Document.filename,
                Document.content_type,
                Document.status,
                Document.created_at,
            )
            .order_by(desc(Document.created_at))
        )
        rows = (await session.execute(statement)).all()

        return [
            DocumentListItem(
                id=str(row.id),
                filename=row.filename,
                content_type=row.content_type,
                status=row.status,
                chunks_count=row.chunks_count,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def _parse_uploaded_content(self, filename: str, content: bytes):
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / filename
            file_path.write_bytes(content)
            return self.parser.parse(file_path)

    async def _mark_document_failed(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> None:
        await session.rollback()
        document = await session.get(Document, document_id)
        if document is None:
            return

        document.status = "failed"
        await session.commit()
