from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.routing.classifier import classify_intent
from app.schemas.chat import ChatRequest

DATASET_PATH = Path(__file__).with_name("intent_dataset.jsonl")


async def main() -> None:
    examples = _load_dataset(DATASET_PATH)
    router_source_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        request = ChatRequest(**example["request"])
        expected_intent = example["expected_intent"]
        result = await classify_intent(request)
        router_source_counts[result.source] += 1

        if result.intent != expected_intent:
            failures.append(
                {
                    "index": index,
                    "question": request.message_text,
                    "expected": expected_intent,
                    "actual": result.intent,
                    "confidence": result.confidence,
                    "source": result.source,
                }
            )

    total = len(examples)
    correct = total - len(failures)
    intent_accuracy = correct / total if total else 0.0
    report = {
        "total": total,
        "correct": correct,
        "intent_accuracy": round(intent_accuracy, 4),
        "router_source_counts": dict(router_source_counts),
        "failed_cases": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                examples.append(json.loads(stripped_line))
    return examples


if __name__ == "__main__":
    asyncio.run(main())

