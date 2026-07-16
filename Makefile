.PHONY: install test lint format typecheck clean build

install:
	pip install -e ".[dev]"
	@echo "Installed agensflow-langgraph + dev deps."

test:
	pytest -v

test-cov:
	pytest -v --cov=src/agensflow_langgraph --cov-report=term --cov-report=html

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src

build:
	hatch build

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
