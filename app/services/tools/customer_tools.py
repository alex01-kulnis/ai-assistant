from __future__ import annotations

from typing import Any


def get_mock_customer_profile(customer_id: str | None) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "segment": "standard",
        "churn_risk": "unknown" if customer_id is None else "medium",
        "open_tickets_count": 0,
        "recommended_action": (
            "Уточнить customer_id для точного анализа."
            if customer_id is None
            else "Проверить последние обращения и предложить релевантное решение."
        ),
    }

