from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from app.agents.support_agent import SYSTEM_PROMPT
from app.core.config import get_settings
from app.schemas.vector import RetrievedChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class LangChainRAGError(RuntimeError):
    pass


class LangChainRAGRetrievalError(LangChainRAGError):
    pass


class LangChainRAGGenerationError(LangChainRAGError):
    pass


class SimilaritySearchWithScore(Protocol):
    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Any, float]]: ...


class AsyncChatModel(Protocol):
    async def ainvoke(self, input: Any) -> Any: ...


@dataclass(frozen=True)
class LangChainRAGResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class LangChainRAGService:
    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        vector_store: SimilaritySearchWithScore | None = None,
        llm: AsyncChatModel | None = None,
        qdrant_client: QdrantClient | None = None,
        collection_name: str | None = None,
        retrieval_limit: int = 5,
    ) -> None:
        settings = get_settings()
        self.embedding_service = embedding_service or get_embedding_service()
        self._vector_store = vector_store
        self._llm = llm
        self._qdrant_client = qdrant_client
        self.qdrant_url = settings.qdrant_url
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.model_name = settings.OLLAMA_MODEL
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.retrieval_limit = retrieval_limit

    async def answer(
        self,
        *,
        question: str,
        intent: str,
        tool_results: str,
    ) -> LangChainRAGResult:
        retrieved_chunks = await self.retrieve(question)
        messages = self._build_prompt_messages(
            question=question,
            intent=intent,
            retrieved_chunks=retrieved_chunks,
            tool_results=tool_results,
        )
        answer = await self._generate(messages)

        return LangChainRAGResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        try:
            async_search = getattr(self.vector_store, "asimilarity_search_with_score", None)
            if async_search is None:
                search_results = self.vector_store.similarity_search_with_score(
                    question,
                    k=self.retrieval_limit,
                )
            else:
                maybe_results = async_search(question, k=self.retrieval_limit)
                search_results = (
                    await maybe_results if inspect.isawaitable(maybe_results) else maybe_results
                )
        except Exception as exc:
            raise LangChainRAGRetrievalError("LangChain Qdrant retrieval failed.") from exc

        return [
            self._to_retrieved_chunk(document=document, score=score)
            for document, score in search_results
        ]

    @property
    def vector_store(self) -> SimilaritySearchWithScore:
        if self._vector_store is None:
            self._vector_store = self._build_vector_store()
        return self._vector_store

    @property
    def llm(self) -> AsyncChatModel:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    @property
    def qdrant_client(self) -> QdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = QdrantClient(
                url=self.qdrant_url,
                check_compatibility=False,
            )
        return self._qdrant_client

    def _build_vector_store(self) -> SimilaritySearchWithScore:
        from langchain_qdrant import QdrantVectorStore

        self._ensure_collection()
        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self._build_langchain_embeddings(),
            content_payload_key="text",
        )

    def _build_langchain_embeddings(self) -> Any:
        from langchain_core.embeddings import Embeddings

        embedding_service = self.embedding_service

        class E5SentenceTransformerEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return embedding_service.embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:

                return embedding_service.embed_query(text)

        return E5SentenceTransformerEmbeddings()

    def _build_llm(self) -> AsyncChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=self.model_name,
            base_url=self.ollama_base_url,
            temperature=0.2,
        )

    def _ensure_collection(self) -> None:
        if self.qdrant_client.collection_exists(collection_name=self.collection_name):
            return

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    def _build_prompt_messages(
        self,
        *,
        question: str,
        intent: str,
        retrieved_chunks: list[RetrievedChunk],
        tool_results: str,
    ) -> Any:
        from langchain_core.prompts import ChatPromptTemplate

        user_prompt = (
            "INTENT:\n{intent}\n\n"
            "CONTEXT:\n{context}\n\n"
            "TOOL RESULTS:\n{tool_results}\n\n"
            "USER QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using only CONTEXT and TOOL RESULTS.\n"
            "- If context is insufficient, answer exactly: "
            "В базе знаний недостаточно информации для ответа на этот вопрос.\n"
            "- Be concise and helpful.\n"
            "- Do not mention filenames, chunk_id, Qdrant, embeddings, or technical details.\n"
            "- Do not add a Sources section."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
        )
        return prompt.format_messages(
            intent=intent,
            context=self._build_context(retrieved_chunks),
            tool_results=tool_results,
            question=question,
        )

    async def _generate(self, messages: Any) -> str:
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:
            raise LangChainRAGGenerationError("LangChain Ollama generation failed.") from exc

        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(self._stringify_content_part(part) for part in content).strip()

        raise LangChainRAGGenerationError("LangChain LLM returned unsupported content.")

    def _to_retrieved_chunk(self, *, document: Any, score: float) -> RetrievedChunk:
        metadata = dict(getattr(document, "metadata", {}) or {})
        print("metadata", metadata)
        point_id = str(
            metadata.get("point_id")
            or metadata.get("_id")
            or metadata.get("id")
            or metadata.get("qdrant_point_id")
            or ""
        )
        payload = self._payload_from_metadata(metadata=metadata, point_id=point_id)
        print("payload", payload)
        page_content = getattr(document, "page_content", "") or payload.get("text", "")

        return RetrievedChunk(
            point_id=point_id,
            score=float(score),
            text=str(page_content),
            document_id=str(payload.get("document_id", "")),
            chunk_id=str(payload.get("chunk_id", "")),
            filename=str(payload.get("filename", "")),
            page_number=payload.get("page_number"),
            chunk_index=int(payload.get("chunk_index", 0)),
        )

    def _payload_from_metadata(self, *, metadata: dict[str, Any], point_id: str) -> dict[str, Any]:
        if self._has_chunk_payload(metadata):
            return metadata
        if not point_id:
            return metadata

        try:
            points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            logger.exception("Failed to retrieve Qdrant payload for point_id=%s", point_id)
            return metadata

        if not points:
            return metadata

        payload = points[0].payload or {}
        return {**metadata, **payload}

    def _has_chunk_payload(self, payload: dict[str, Any]) -> bool:
        return all(key in payload for key in ("document_id", "chunk_id", "filename", "chunk_index"))

    def _build_context(self, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "No relevant context was found."

        context_parts: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            page_label = chunk.page_number if chunk.page_number is not None else "n/a"
            context_parts.append(f"[Context fragment {index}, page={page_label}]\n{chunk.text}")

        return "\n\n".join(context_parts)

    def _stringify_content_part(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                return text
        return str(part)


@lru_cache
def get_langchain_rag_service() -> LangChainRAGService:
    return LangChainRAGService()
