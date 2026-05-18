import json

import httpx
import pytest

from app.integrations.telegram.formatters import (
    format_answer,
    format_sources,
    remove_source_mentions,
)
from app.integrations.telegram.service import (
    TelegramChatClient,
    is_user_allowed,
    parse_allowed_user_ids,
)


def test_parse_allowed_user_ids() -> None:
    assert parse_allowed_user_ids("123456, 789012") == {123456, 789012}


def test_access_allowed_when_allowed_user_ids_is_empty() -> None:
    assert parse_allowed_user_ids(None) is None
    assert parse_allowed_user_ids("") is None
    assert is_user_allowed(123456, None) is True


def test_access_denied_for_user_not_in_list() -> None:
    allowed_user_ids = parse_allowed_user_ids("123456,789012")

    assert allowed_user_ids is not None
    assert is_user_allowed(111111, allowed_user_ids) is False


def test_format_sources_formats_detailed_sources() -> None:
    sources = [
        {
            "filename": "refund_policy.txt",
            "page_number": None,
            "chunk_index": 0,
            "score": 0.8872,
        }
    ]

    assert format_sources(sources) == "1. refund_policy.txt | page: - | chunk: 0 | score: 0.887"


def test_format_answer_does_not_return_old_sources_hint() -> None:
    formatted_answer = format_answer(
        "Возврат можно оформить в течение 14 дней.",
        sources=[{"filename": "refund_policy.txt"}],
    )

    assert "Источники: /sources" not in formatted_answer


def test_format_answer_adds_details_command_when_sources_exist() -> None:
    formatted_answer = format_answer(
        "Возврат можно оформить в течение 14 дней.",
        sources=[
            {"filename": "refund_policy.txt"},
            {"filename": "refund_policy.txt"},
            {"filename": "payments.txt"},
            {"filename": "orders.txt"},
            {"filename": "extra.txt"},
        ],
    )

    assert "📎 Источники: refund_policy.txt, payments.txt, orders.txt" in formatted_answer
    assert "Подробнее: /sources" in formatted_answer
    assert "extra.txt" not in formatted_answer


def test_remove_source_mentions_removes_file_source_line() -> None:
    cleaned_answer = remove_source_mentions(
        "Возврат можно оформить в течение 14 дней.\n"
        "Информация взята из файла refund_policy.txt."
    )

    assert cleaned_answer == "Возврат можно оформить в течение 14 дней."


def test_remove_source_mentions_does_not_break_regular_text() -> None:
    answer = "Для возврата укажите причину обращения и номер заказа."

    assert remove_source_mentions(answer) == answer


def test_format_answer_truncates_long_response() -> None:
    formatted_answer = format_answer("a" * 5000, sources=[])

    assert len(formatted_answer) <= 4096
    assert "Ответ обрезан" in formatted_answer


@pytest.mark.asyncio
async def test_telegram_chat_client_forms_http_payload() -> None:
    captured_payload: dict | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "conversation_id": "conversation-1",
                "message_id": "message-1",
                "answer": "Ответ",
                "sources": [
                    {
                        "document_id": "document-1",
                        "filename": "example.txt",
                        "page_number": None,
                        "chunk_index": 0,
                        "score": 0.9,
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        chat_client = TelegramChatClient(
            use_backend_http=True,
            backend_chat_url="http://backend.test/api/v1/chat",
            http_client=http_client,
        )

        response = await chat_client.ask(
            message="Как оформить возврат?",
            conversation_id="conversation-0",
        )

    assert captured_payload == {
        "conversation_id": "conversation-0",
        "message": "Как оформить возврат?",
    }
    assert response.conversation_id == "conversation-1"
    assert response.answer == "Ответ"
    assert response.sources[0]["filename"] == "example.txt"
