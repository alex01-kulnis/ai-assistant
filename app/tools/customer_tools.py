from __future__ import annotations

from typing import Any


async def get_mock_customer_profile(customer_id: str | None) -> dict[str, Any]:
    if customer_id is None:
        return {
            "customer_id": None,
            "error": "missing_customer_id",
        }

    if customer_id == "123":
        return {
            "customer_id": "123",
            "segment": "premium",
            "lifetime_value": 15200,
            "last_purchase_days_ago": 45,
            "support_tickets_30d": 3,
            "email_open_rate_30d": 0.05,
            "last_login_days_ago": 21,
            "recent_complaints": 2,
        }

    return {
        "customer_id": customer_id,
        "segment": "standard",
        "lifetime_value": 2400,
        "last_purchase_days_ago": 12,
        "support_tickets_30d": 1,
        "email_open_rate_30d": 0.24,
        "last_login_days_ago": 5,
        "recent_complaints": 0,
    }


def calculate_churn_score(profile: dict[str, Any]) -> dict[str, Any]:
    if profile.get("error") == "missing_customer_id":
        return {
            "score": 0.0,
            "risk_level": "unknown",
            "signals": [],
        }

    score = 0.0
    signals: list[str] = []
    if profile.get("last_purchase_days_ago", 0) > 30:
        score += 0.30
        signals.append("last_purchase_days_ago > 30")
    if profile.get("support_tickets_30d", 0) >= 3:
        score += 0.25
        signals.append("support_tickets_30d >= 3")
    if profile.get("email_open_rate_30d", 1.0) < 0.10:
        score += 0.20
        signals.append("email_open_rate_30d < 0.10")
    if profile.get("last_login_days_ago", 0) > 14:
        score += 0.25
        signals.append("last_login_days_ago > 14")
    if profile.get("recent_complaints", 0) >= 2:
        score += 0.15
        signals.append("recent_complaints >= 2")

    score = min(score, 1.0)
    if score >= 0.70:
        risk_level = "high"
    elif score >= 0.40:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "score": round(score, 2),
        "risk_level": risk_level,
        "signals": signals,
    }


def recommend_next_best_action(
    profile: dict[str, Any],
    churn_score: dict[str, Any],
) -> dict[str, Any]:
    if profile.get("error") == "missing_customer_id":
        return {
            "recommendation": "need_customer_id",
            "action": "request_customer_id",
            "channel": None,
            "reason": "customer_id is required for customer-specific analysis",
        }

    risk_level = churn_score.get("risk_level")
    if risk_level == "high":
        return {
            "recommendation": "reactivation_offer_or_personal_contact",
            "action": "reactivation_offer_or_personal_contact",
            "channel": "personal_call_or_email",
            "reason": "high churn risk signals",
        }
    if risk_level == "medium":
        return {
            "recommendation": "targeted_retention_message",
            "action": "targeted_retention_message",
            "channel": "email_or_push",
            "reason": "moderate churn risk signals",
        }
    return {
        "recommendation": "regular_engagement",
        "action": "regular_engagement",
        "channel": "standard_campaign",
        "reason": "low churn risk signals",
    }

