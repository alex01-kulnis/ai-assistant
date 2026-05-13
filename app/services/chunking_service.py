from __future__ import annotations

from dataclasses import dataclass

from app.services.document_parser import ParsedPage


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    page_number: int | None


class TextChunkingService:
    def split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> list[str]:
        self._validate_params(chunk_size=chunk_size, overlap=overlap)

        if not text:
            return []

        if len(text) <= chunk_size:
            stripped_text = text.strip()
            return [stripped_text] if stripped_text else []

        chunks: list[str] = []
        step = chunk_size - overlap
        start = 0

        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if start + chunk_size >= len(text):
                break

            start += step

        return chunks

    def split_pages(
        self,
        pages: list[ParsedPage],
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> list[Chunk]:
        self._validate_params(chunk_size=chunk_size, overlap=overlap)

        chunks: list[Chunk] = []
        for page in pages:
            page_chunks = self.split_text(
                page.text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            for chunk_text in page_chunks:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                    )
                )

        return chunks

    def _validate_params(self, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
