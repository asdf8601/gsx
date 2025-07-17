# Makefile

# Objetivos
.PHONY: uv
uv:  ## install uv
	@command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

.PHONY: install
install: uv
	uv sync

.PHONY: dev
dev: uv
	uv sync --dev

.PHONY: lint
lint: dev
	uv run ruff check src/

.PHONY: fix
fix: dev
	uv run ruff check src/ --fix

.PHONY: test
test: dev
	uv run pytest tests/

.PHONY: install-precommit
install-precommit:
	uv run pre-commit install

.PHONY: ci
ci:
	uv run pre-commit run --all-files

.PHONY: tag
tag:  ## Create a new tag and modify files
	@if [ -z "$(v)" ]; then echo "Usage: make tag v=<version>\ncurrent tag: $(shell grep version pyproject.toml | tr -d '"')"; exit 1;fi
	sed -i 's/version = ".*"/version = "$(v)"/g' pyproject.toml
	sed -i 's/__version__ = ".*"/__version__ = "$(v)"/g' src/gsx.py
	git add pyproject.toml src/gsx.py
	git commit -m "New version $(v)"
	git tag "$(v)"
