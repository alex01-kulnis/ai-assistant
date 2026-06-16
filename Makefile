PYTHON ?= .venv/bin/python

.PHONY: dev docker-up docker-down migrate test lint telegram-bot eval-intents eval-agents

dev:
	$(PYTHON) -m uvicorn app.main:app --reload

docker-up:
	docker compose up -d

docker-down:
	docker compose down

migrate:
	$(PYTHON) -m alembic upgrade head

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check app tests

telegram-bot:
	$(PYTHON) scripts/run_telegram_bot.py

eval-intents:
	$(PYTHON) -m app.evaluation.evaluate_intent_routing

eval-agents:
	$(PYTHON) -m app.evaluation.evaluate_multi_agent
