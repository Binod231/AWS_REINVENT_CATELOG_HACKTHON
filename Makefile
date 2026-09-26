.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv
RUN := $(UV) run
OUT ?= out

.PHONY: help install lint fmt types test check demo clean hooks

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install everything (locked)
	$(UV) sync --extra all

hooks: install ## Install git pre-commit hooks
	$(RUN) pre-commit install

lint: ## Ruff lint
	$(RUN) ruff check .

fmt: ## Ruff format (writes)
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

types: ## Mypy strict
	$(RUN) mypy

test: ## Pytest
	$(RUN) pytest

check: lint types test ## Everything CI runs

demo: ## End-to-end demo on bundled fixtures. No credentials, no AWS account.
	@bash scripts/demo.sh

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage $(OUT)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
