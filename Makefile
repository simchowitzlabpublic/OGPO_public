.PHONY: sync
sync:
	uv sync --extra cuda12 --extra robomimic

.PHONY: sync-cuda13
sync-cuda13:
	uv sync --extra cuda13 --extra robomimic

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run pyright

.PHONY: check
check: format type
