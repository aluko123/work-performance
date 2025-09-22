SHELL := /bin/bash

.PHONY: help setup dev front docker-up docker-down smoke

help:
	@echo "Targets:"
	@echo "  setup       Create venv and install requirements"
	@echo "  dev         Run backend (uvicorn --reload on :8000)"
	@echo "  front       Serve frontend on :8001"
	@echo "  docker-up   Build and start backend+frontend"
	@echo "  docker-down Stop containers"
	@echo "  smoke       Basic API curl checks"
	@echo "  test        Run pytest offline"
	@echo "  test-cov    Run pytest with coverage report"

setup:
	python3 -m venv .venv
	@if [ -f ".venv/bin/pip" ]; then PIP=.venv/bin/pip; else PIP=.venv/Scripts/pip.exe; fi; \
		$$PIP install -r requirements.txt

dev:
	@if [ -f ".venv/bin/python" ]; then PY=.venv/bin/python; else PY=python; fi; \
		$$PY -m uvicorn backend.main:app --reload --port 8000

front:
	python -m http.server 8001 -d frontend

docker-up:
	docker compose up --build

docker-down:
	docker compose down

smoke:
	@echo "GET /analyses/" && curl -sS http://localhost:8000/analyses/ | head -c 200 || true
	@echo "\nGET /api/trends (comm_clarity,daily)" && curl -sS "http://localhost:8000/api/trends?metric=comm_clarity&period=daily" | head -c 200 || true
	@echo "\nPOST /api/get_insights" && curl -sS -H 'Content-Type: application/json' -d '{"question":"What improved last week?"}' http://localhost:8000/api/get_insights | head -c 200 || true

test:
	pytest -q backend/tests

test-cov:
	pytest -q --cov=backend --cov-report=term-missing backend/tests
