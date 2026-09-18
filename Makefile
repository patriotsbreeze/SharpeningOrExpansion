.PHONY: help install test smoke lint clean figures
SHELL := /bin/bash
export PYTHONPATH := src
ROOT ?= runs
PY := .venv/bin/python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

install: ## Create the venv and install CPU dependencies
	uv venv --python 3.11 .venv
	uv pip install --python .venv/bin/python -e ".[dev]"

test: ## Run the full CPU test suite
	$(PY) -m pytest -q

lint: ## ruff check + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

smoke: ## Full pipeline on the mock backend: no GPU, no network
	rm -rf $(ROOT)/exp=smoke_mock
	$(PY) -m soe.cli prepare  configs/experiments/smoke_mock.yaml --root $(ROOT)
	$(PY) -m soe.cli plan     configs/experiments/smoke_mock.yaml --root $(ROOT)
	$(PY) -m soe.cli generate configs/experiments/smoke_mock.yaml --root $(ROOT) --worker 0
	$(PY) -m soe.cli grade    configs/experiments/smoke_mock.yaml --root $(ROOT) --graders fastint
	$(PY) -m soe.cli verify   configs/experiments/smoke_mock.yaml --root $(ROOT)
	$(PY) -m soe.cli figures  configs/experiments/smoke_mock.yaml --root $(ROOT) \
		--base mock_base --rl mock_rl --n-boot 400

figures: ## Regenerate figures for a completed experiment: make figures CONFIG=... BASE=... RL=...
	$(PY) -m soe.cli figures $(CONFIG) --root $(ROOT) --base $(BASE) --rl $(RL)

clean:
	rm -rf .pytest_cache .ruff_cache runs/exp=smoke_mock
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
