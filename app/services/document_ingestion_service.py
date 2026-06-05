from __future__ import annotations

import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tracing import set_span_attributes
from app.models.document import Document, DocumentChunk
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListItem,
    DocumentReindexResponse,
    DocumentUploadResponse,
)
from app.schemas.vector import VectorChunkInput
from app.services.chunking_service import Chunk, TextChunkingService
from app.services.document_parser import DocumentParser, DocumentParsingError, ParsedDocument
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.vectorstore.qdrant_store import QdrantVectorStore, get_qdrant_vector_store

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class DocumentIndexingError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class IndexingSpanNames:
    parse: str
    chunk: str
    embed: str
    upsert: str


@dataclass(frozen=True)
class IndexedDocumentData:
    parsed_document: ParsedDocument
    chunks: list[Chunk]
    chunk_ids: list[uuid.UUID]
    qdrant_point_ids: list[str]


UPLOAD_INDEXING_SPANS = IndexingSpanNames(
    parse="document.parse",
    chunk="chunking.split_pages",
    embed="embedding.embed_documents",
    upsert="qdrant.upsert_chunks",
)

REINDEX_INDEXING_SPANS = IndexingSpanNames(
    parse="document.reindex.parse",
    chunk="document.reindex.chunk",
    embed="document.reindex.embed",
    upsert="qdrant.upsert_new_points",
)


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
        with tracer.start_as_current_span("document_ingestion.index_uploaded_document") as span:
            set_span_attributes(
                span,
                {
                    "filename": safe_filename,
                    "content_type": content_type or "application/octet-stream",
                },
            )
            return await self._index_uploaded_document_traced(
                safe_filename=safe_filename,
                content_type=content_type,
                content=content,
                session=session,
            )

    async def delete_document(
        self,
        *,
        document_id: str,
        session: AsyncSession,
    ) -> DocumentDeleteResponse:
        parsed_document_id = self._parse_document_id(document_id)
        with tracer.start_as_current_span("document.delete") as span:
            set_span_attributes(span, {"document_id": str(parsed_document_id)})

            document: Document | None = None
            chunks: list[DocumentChunk] = []
            deleted_qdrant_points_count = 0
            try:
                with tracer.start_as_current_span("document.delete.load_document") as load_span:
                    document = await session.get(Document, parsed_document_id)
                    if document is None:
                        raise DocumentIndexingError("Document not found.", status_code=404)
                    set_span_attributes(
                        load_span,
                        {
                            "document_id": str(document.id),
                            "filename": document.filename,
                            "content_type": document.content_type,
                            "status": document.status,
                        },
                    )

                with tracer.start_as_current_span("document.delete.load_chunks") as chunks_span:
                    chunks = await self._load_document_chunks(
                        session=session,
                        document_id=parsed_document_id,
                    )
                    set_span_attributes(chunks_span, {"chunks_count": len(chunks)})

                with tracer.start_as_current_span("qdrant.delete_points") as qdrant_span:
                    deleted_qdrant_points_count = self._delete_qdrant_points_for_document(
                        document_id=parsed_document_id,
                        chunks=chunks,
                    )
                    set_span_attributes(
                        qdrant_span,
                        {
                            "document_id": str(parsed_document_id),
                            "qdrant_points_count": deleted_qdrant_points_count,
                        },
                    )

                with tracer.start_as_current_span("db.document_chunks.delete") as db_chunks_span:
                    await self._delete_document_chunks(session=session, chunks=chunks)
                    set_span_attributes(db_chunks_span, {"chunks_count": len(chunks)})

                with tracer.start_as_current_span("db.document.delete") as db_document_span:
                    filename = document.filename
                    await session.delete(document)
                    await session.commit()
                    set_span_attributes(
                        db_document_span,
                        {
                            "document_id": str(parsed_document_id),
                            "filename": filename,
                            "status": "deleted",
                        },
                    )

                return DocumentDeleteResponse(
                    document_id=str(parsed_document_id),
                    filename=filename,
                    deleted_chunks_count=len(chunks),
                    deleted_qdrant_points_count=deleted_qdrant_points_count,
                    status="deleted",
                )
            except DocumentIndexingError:
                raise
            except Exception as exc:
                logger.exception("Document deletion failed for document_id=%s", document_id)
                await session.rollback()
                await self._try_mark_existing_document_failed(
                    session=session,
                    document_id=parsed_document_id,
                )
                raise DocumentIndexingError("Failed to delete document.", status_code=500) from exc

    async def reindex_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str | None,
        content: bytes,
        session: AsyncSession,
    ) -> DocumentReindexResponse:
        parsed_document_id = self._parse_document_id(document_id)
        safe_filename = Path(filename).name
        if not safe_filename:
            raise DocumentIndexingError(
                "Reindex requires a new uploaded file because original files are not persisted.",
                status_code=400,
            )

        with tracer.start_as_current_span("document.reindex") as span:
            set_span_attributes(
                span,
                {
                    "document_id": str(parsed_document_id),
                    "filename": safe_filename,
                    "content_type": content_type or "application/octet-stream",
                },
            )

            indexed_document_data: IndexedDocumentData | None = None
            try:
                document = await session.get(Document, parsed_document_id)
                if document is None:
                    raise DocumentIndexingError("Document not found.", status_code=404)

                document.status = "processing"
                await session.commit()

                with tracer.start_as_current_span(
                    "document.reindex.cleanup_old_chunks"
                ) as cleanup_span:
                    old_chunks = await self._load_document_chunks(
                        session=session,
                        document_id=parsed_document_id,
                    )
                    set_span_attributes(cleanup_span, {"chunks_count": len(old_chunks)})

                    with tracer.start_as_current_span("qdrant.delete_old_points") as qdrant_span:
                        deleted_qdrant_points_count = self._delete_qdrant_points_for_document(
                            document_id=parsed_document_id,
                            chunks=old_chunks,
                        )
                        set_span_attributes(
                            qdrant_span,
                            {
                                "document_id": str(parsed_document_id),
                                "qdrant_points_count": deleted_qdrant_points_count,
                            },
                        )

                    await self._delete_document_chunks(session=session, chunks=old_chunks)
                    document.filename = safe_filename
                    document.content_type = content_type or "application/octet-stream"
                    document.status = "processing"
                    await session.commit()

                indexed_document_data = self._build_indexed_document_data(
                    document=document,
                    filename=safe_filename,
                    content=content,
                    span_names=REINDEX_INDEXING_SPANS,
                )
                document.content_type = indexed_document_data.parsed_document.content_type

                with tracer.start_as_current_span("db.document_chunks.save") as save_span:
                    self._add_document_chunks(
                        session=session,
                        document=document,
                        indexed_document_data=indexed_document_data,
                    )
                    set_span_attributes(
                        save_span,
                        {"chunks_count": len(indexed_document_data.chunks)},
                    )

                with tracer.start_as_current_span("document.reindex.mark_indexed") as indexed_span:
                    document.status = "indexed"
                    await session.commit()
                    set_span_attributes(
                        indexed_span,
                        {
                            "document_id": str(document.id),
                            "filename": document.filename,
                            "content_type": document.content_type,
                            "status": document.status,
                            "chunks_count": len(indexed_document_data.chunks),
                        },
                    )

                return DocumentReindexResponse(
                    document_id=str(document.id),
                    filename=document.filename,
                    chunks_count=len(indexed_document_data.chunks),
                    status=document.status,
                )
            except DocumentIndexingError:
                await self._cleanup_new_qdrant_points(indexed_document_data)
                await self._mark_document_failed(
                    session=session,
                    document_id=parsed_document_id,
                    span_name="document.reindex.mark_failed",
                )
                raise
            except (DocumentParsingError, UnicodeDecodeError) as exc:
                await self._cleanup_new_qdrant_points(indexed_document_data)
                await self._mark_document_failed(
                    session=session,
                    document_id=parsed_document_id,
                    span_name="document.reindex.mark_failed",
                )
                raise DocumentIndexingError(str(exc), status_code=400) from exc
            except Exception as exc:
                logger.exception("Document reindex failed for document_id=%s", document_id)
                await self._cleanup_new_qdrant_points(indexed_document_data)
                await self._mark_document_failed(
                    session=session,
                    document_id=parsed_document_id,
                    span_name="document.reindex.mark_failed",
                )
                raise DocumentIndexingError(
                    "Failed to reindex document. Old chunks were removed in replace mode.",
                    status_code=500,
                ) from exc

    async def _index_uploaded_document_traced(
        self,
        *,
        safe_filename: str,
        content_type: str | None,
        content: bytes,
        session: AsyncSession,
    ) -> DocumentUploadResponse:
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

        indexed_document_data: IndexedDocumentData | None = None
        try:
            indexed_document_data = self._build_indexed_document_data(
                document=document,
                filename=safe_filename,
                content=content,
                span_names=UPLOAD_INDEXING_SPANS,
            )
            document.content_type = indexed_document_data.parsed_document.content_type

            with tracer.start_as_current_span("db.save_document_chunks") as span:
                self._add_document_chunks(
                    session=session,
                    document=document,
                    indexed_document_data=indexed_document_data,
                )
                set_span_attributes(
                    span,
                    {"chunks_count": len(indexed_document_data.chunks)},
                )

            with tracer.start_as_current_span("document.mark_indexed") as span:
                document.status = "indexed"
                await session.commit()
                set_span_attributes(
                    span,
                    {
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "status": document.status,
                        "chunks_count": len(indexed_document_data.chunks),
                    },
                )

            return DocumentUploadResponse(
                document_id=str(document.id),
                filename=document.filename,
                chunks_count=len(indexed_document_data.chunks),
                status=document.status,
            )
        except DocumentIndexingError as exc:
            await self._cleanup_new_qdrant_points(indexed_document_data)
            await self._mark_document_failed(session=session, document_id=document.id)
            raise exc
        except (DocumentParsingError, UnicodeDecodeError) as exc:
            await self._cleanup_new_qdrant_points(indexed_document_data)
            await self._mark_document_failed(session=session, document_id=document.id)
            raise DocumentIndexingError(str(exc), status_code=400) from exc
        except Exception as exc:
            logger.exception("Document indexing failed for document_id=%s", document.id)
            await self._cleanup_new_qdrant_points(indexed_document_data)
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

    def _build_indexed_document_data(
        self,
        *,
        document: Document,
        filename: str,
        content: bytes,
        span_names: IndexingSpanNames,
    ) -> IndexedDocumentData:
        with tracer.start_as_current_span(span_names.parse) as span:
            parsed_document = self._parse_uploaded_content(
                filename=filename,
                content=content,
            )
            set_span_attributes(
                span,
                {
                    "filename": parsed_document.filename,
                    "content_type": parsed_document.content_type,
                },
            )

        with tracer.start_as_current_span(span_names.chunk) as span:
            chunks = self.chunking_service.split_pages(parsed_document.pages)
            set_span_attributes(span, {"chunks_count": len(chunks)})
        if not chunks:
            raise DocumentIndexingError(
                "Document does not contain text to index.",
                status_code=400,
            )

        with tracer.start_as_current_span(span_names.embed) as span:
            embeddings = self.embedding_service.embed_documents([chunk.text for chunk in chunks])
            set_span_attributes(
                span,
                {
                    "chunks_count": len(chunks),
                    "vector_size": len(embeddings[0]) if embeddings else None,
                },
            )

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

        with tracer.start_as_current_span(span_names.upsert) as span:
            qdrant_point_ids = self.vector_store.upsert_chunks(vector_chunks)
            set_span_attributes(span, {"chunks_count": len(vector_chunks)})

        return IndexedDocumentData(
            parsed_document=parsed_document,
            chunks=chunks,
            chunk_ids=chunk_ids,
            qdrant_point_ids=qdrant_point_ids,
        )

    def _add_document_chunks(
        self,
        *,
        session: AsyncSession,
        document: Document,
        indexed_document_data: IndexedDocumentData,
    ) -> None:
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
                    indexed_document_data.chunks,
                    indexed_document_data.chunk_ids,
                    indexed_document_data.qdrant_point_ids,
                    strict=True,
                )
            ]
        )

    async def _load_document_chunks(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> list[DocumentChunk]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        result = await session.execute(statement)
        return list(result.scalars().all())

    async def _delete_document_chunks(
        self,
        *,
        session: AsyncSession,
        chunks: list[DocumentChunk],
    ) -> None:
        for chunk in chunks:
            await session.delete(chunk)

    def _delete_qdrant_points_for_document(
        self,
        *,
        document_id: uuid.UUID,
        chunks: list[DocumentChunk],
    ) -> int:
        point_ids = [chunk.qdrant_point_id for chunk in chunks if chunk.qdrant_point_id]
        deleted_points_count = self.vector_store.delete_points(point_ids)
        deleted_points_count += self.vector_store.delete_points_by_document_id(str(document_id))
        return deleted_points_count

    async def _cleanup_new_qdrant_points(
        self,
        indexed_document_data: IndexedDocumentData | None,
    ) -> None:
        if indexed_document_data is None or not indexed_document_data.qdrant_point_ids:
            return

        try:
            self.vector_store.delete_points(indexed_document_data.qdrant_point_ids)
        except Exception:
            logger.exception("Failed to cleanup newly upserted Qdrant points")

    def _parse_uploaded_content(self, filename: str, content: bytes) -> ParsedDocument:
        with tempfile.TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / filename
            file_path.write_bytes(content)
            return self.parser.parse(file_path)

    def _parse_document_id(self, document_id: str) -> uuid.UUID:
        try:
            return uuid.UUID(document_id)
        except ValueError as exc:
            raise DocumentIndexingError(
                "document_id must be a valid UUID.",
                status_code=400,
            ) from exc

    async def _try_mark_existing_document_failed(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
    ) -> None:
        try:
            await self._mark_document_failed(session=session, document_id=document_id)
        except Exception:
            logger.exception("Failed to mark document failed after lifecycle error")

    async def _mark_document_failed(
        self,
        *,
        session: AsyncSession,
        document_id: uuid.UUID,
        span_name: str = "document.mark_failed",
    ) -> None:
        with tracer.start_as_current_span(span_name) as span:
            await session.rollback()
            document = await session.get(Document, document_id)
            if document is None:
                set_span_attributes(span, {"status": "failed"})
                return

            document.status = "failed"
            await session.commit()
            set_span_attributes(
                span,
                {
                    "filename": document.filename,
                    "content_type": document.content_type,
                    "status": document.status,
                },
            )
