from __future__ import annotations

UNSUPPORTED_REQUEST_MESSAGE = (
    "Я не могу надежно обработать этот запрос в текущем сценарии. "
    "Попробуйте задать вопрос по документам, тикету или клиентскому профилю."
)

INSUFFICIENT_SUMMARY_TEXT_MESSAGE = (
    "Недостаточно текста для суммаризации. "
    "Передайте текст через selected_text или укажите ticket_id."
)


def build_fallback_answer() -> str:
    return UNSUPPORTED_REQUEST_MESSAGE

