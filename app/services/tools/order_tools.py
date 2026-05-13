from __future__ import annotations

import hashlib


def get_order_status(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "status": "in_progress",
        "message": "Order status is available in the customer account.",
    }


def get_refund_status(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "refund_status": "processing",
        "message": "Refund requests are reviewed by the support team.",
    }


def create_support_ticket(reason: str, user_message: str) -> dict:
    ticket_hash = hashlib.sha1(f"{reason}:{user_message}".encode("utf-8")).hexdigest()[:10]
    return {
        "ticket_id": f"TICKET-{ticket_hash.upper()}",
        "reason": reason,
        "status": "created",
    }
