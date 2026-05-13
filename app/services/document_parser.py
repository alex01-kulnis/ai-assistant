from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PDF_NO_TEXT_ERROR = "PDF does not contain extractable text. OCR is not supported yet."


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    content_type: str
    pages: list[ParsedPage]
    full_text: str


class DocumentParsingError(ValueError):
    pass


class UnsupportedDocumentTypeError(DocumentParsingError):
    pass


class DocumentParser:
    _content_types = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".pdf": "application/pdf",
    }

    def parse(self, file_path: str | Path) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix not in self._content_types:
            unsupported_extension = suffix or "<none>"
            raise UnsupportedDocumentTypeError(
                f"Unsupported document extension: {unsupported_extension}"
            )

        if suffix in {".txt", ".md"}:
            return self._parse_text_file(path)

        return self._parse_pdf_file(path)

    def _parse_text_file(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8")
        page = ParsedPage(page_number=None, text=text)
        return ParsedDocument(
            filename=path.name,
            content_type=self._content_types[path.suffix.lower()],
            pages=[page],
            full_text=text,
        )

    def _parse_pdf_file(self, path: Path) -> ParsedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParsingError("PyMuPDF is required to parse PDF documents.") from exc

        pages: list[ParsedPage] = []

        with fitz.open(path) as pdf_document:
            for page_index, page in enumerate(pdf_document, start=1):
                text = page.get_text("text")
                pages.append(ParsedPage(page_number=page_index, text=text))

        full_text = "\n".join(page.text for page in pages)
        if not full_text.strip():
            raise DocumentParsingError(PDF_NO_TEXT_ERROR)

        return ParsedDocument(
            filename=path.name,
            content_type=self._content_types[".pdf"],
            pages=pages,
            full_text=full_text,
        )
