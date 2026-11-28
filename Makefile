# Uses .venv/bin/python if a local venv exists, otherwise python3 (e.g. on Colab).
# Override with: make install PYTHON=/path/to/python
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: help venv install test lint format api web

help:
	@echo "make venv     create .venv (local laptops)"
	@echo "make install  pip install -e '.[api,dev]'"
	@echo "make test     run pytest"
	@echo "make lint     ruff check"
	@echo "make format   ruff format + import sort"
	@echo "make api      backend on :8000 with reload"
	@echo "make web      frontend dev server on :5173"

venv:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip

install:
	$(PYTHON) -m pip install -e '.[api,dev]'

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --select I --fix .
	$(PYTHON) -m ruff format .

api:
	$(PYTHON) -m uvicorn app.main:app --app-dir backend --reload --port 8000

web:
	cd frontend && $(if $(wildcard frontend/node_modules),,npm install && )npm run dev
