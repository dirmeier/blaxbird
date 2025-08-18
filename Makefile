.PHONY: tests, lints, docs, format

tests:
	uv run pytest

lints:
	uv run ruff check blaxbird examples

format:
	uv run ruff check --select I --fix blaxbird examples
	uv run ruff format blaxbird examples
