from __future__ import annotations

import re
from typing import Any

TELEGRAM_MESSAGE_LIMIT = 4096
TRUNCATION_SUFFIX = "\n\n[Ответ обрезан из-за лимита Telegram.]"
SOURCE_MENTION_PATTERNS = (
    re.compile(r"^\s*Информация взята из файла\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Источник\s*:\s*.*$", re.IGNORECASE),
    re.compile(r"^\s*Источники\s*:\s*.*$", re.IGNORECASE),
)


def format_answer(answer: str, sources: list[Any]) -> str:
    cleaned_answer = remove_source_mentions(answer)
    source_filenames = _unique_source_filenames(sources, limit=3)
    if not source_filenames:
        return truncate_telegram_message(cleaned_answer)

    source_label = "Источник" if len(source_filenames) == 1 else "Источники"
    source_hint = f"\n\n📎 {source_label}: {', '.join(source_filenames)}\nПодробнее: /sources"
    return truncate_telegram_message(f"{cleaned_answer}{source_hint}")


def format_sources(sources: list[Any]) -> str:
    if not sources:
        return "Источники для последнего ответа не найдены."

    lines: list[str] = []
    for index, source in enumerate(sources, start=1):
        filename = _get_source_value(source, "filename", "-")
        page_number = _get_source_value(source, "page_number", None)
        chunk_index = _get_source_value(source, "chunk_index", "-")
        score = _get_source_value(source, "score", None)

        page_label = page_number if page_number is not None else "-"
        score_label = f"{float(score):.3f}" if score is not None else "-"
        lines.append(
            f"{index}. {filename} | page: {page_label} | "
            f"chunk: {chunk_index} | score: {score_label}"
        )

    return truncate_telegram_message("\n".join(lines))


def remove_source_mentions(answer: str) -> str:
    lines = answer.splitlines()
    cleaned_lines = [
        line
        for line in lines
        if not any(pattern.match(line) for pattern in SOURCE_MENTION_PATTERNS)
    ]
    return "\n".join(cleaned_lines).strip()


def truncate_telegram_message(message: str) -> str:
    if len(message) <= TELEGRAM_MESSAGE_LIMIT:
        return message

    max_body_length = TELEGRAM_MESSAGE_LIMIT - len(TRUNCATION_SUFFIX)
    return f"{message[:max_body_length].rstrip()}{TRUNCATION_SUFFIX}"


def _get_source_value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _unique_source_filenames(sources: list[Any], limit: int) -> list[str]:
    filenames: list[str] = []
    seen_filenames: set[str] = set()
    for source in sources:
        filename = str(_get_source_value(source, "filename", "")).strip()
        if not filename or filename in seen_filenames:
            continue

        filenames.append(filename)
        seen_filenames.add(filename)
        if len(filenames) >= limit:
            break

    return filenames
