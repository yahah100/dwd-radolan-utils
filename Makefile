.PHONY: help install test lint format type-check coverage clean build docs

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync --dev

install-hooks: ## Install pre-commit hooks
	uv run pre-commit install

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=dwd_radolan_utils --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without slow integration tests
	uv run pytest -m "not slow"

lint: ## Run linting
	uv run ruff check src tests

lint-fix: ## Run linting with auto-fix
	uv run ruff check --fix src tests

format: ## Format code
	uv run ruff format src tests

format-check: ## Check code formatting
	uv run ruff format --check src tests

type-check: ## Run type checking
	uv run mypy src

security: ## Run security checks
	uv run bandit -r src/ || true
	uv export --dev --format requirements-txt | uv run safety check --stdin || true

coverage: ## Generate coverage report
	uv run coverage run -m pytest
	uv run coverage report --show-missing
	uv run coverage html

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +

build: ## Build package
	uv build

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

all-checks: lint format-check type-check security test ## Run all quality checks

ci: install all-checks ## Run CI pipeline locally

.DEFAULT_GOAL := help 