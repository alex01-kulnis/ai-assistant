from pathlib import Path

import pytest

from app.services.document_parser import DocumentParser, UnsupportedDocumentTypeError


def test_parse_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "support.txt"
    file_path.write_text("Hello support\nSecond line", encoding="utf-8")

    parsed_document = DocumentParser().parse(file_path)

    assert parsed_document.filename == "support.txt"
    assert parsed_document.content_type == "text/plain"
    assert parsed_document.full_text == "Hello support\nSecond line"
    assert len(parsed_document.pages) == 1
    assert parsed_document.pages[0].page_number is None
    assert parsed_document.pages[0].text == "Hello support\nSecond line"


def test_parse_md_file(tmp_path: Path) -> None:
    file_path = tmp_path / "faq.md"
    file_path.write_text("# FAQ\n\nReset your password from settings.", encoding="utf-8")

    parsed_document = DocumentParser().parse(file_path)

    assert parsed_document.filename == "faq.md"
    assert parsed_document.content_type == "text/markdown"
    assert parsed_document.full_text == "# FAQ\n\nReset your password from settings."
    assert len(parsed_document.pages) == 1
    assert parsed_document.pages[0].page_number is None


def test_parse_empty_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    parsed_document = DocumentParser().parse(file_path)

    assert parsed_document.filename == "empty.txt"
    assert parsed_document.content_type == "text/plain"
    assert parsed_document.full_text == ""
    assert len(parsed_document.pages) == 1
    assert parsed_document.pages[0].text == ""


def test_parse_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "archive.docx"
    file_path.write_text("not supported", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentTypeError, match="Unsupported document extension"):
        DocumentParser().parse(file_path)
