.PHONY: dev dev-build test test-unit test-integration test-e2e test-security \
        lint format typecheck clean migrate logs

# ── Local Development ─────────────────────────────────────────────────────────

dev:
	cd infrastructure/compose && docker compose up

dev-build:
	cd infrastructure/compose && docker compose up --build

dev-down:
	cd infrastructure/compose && docker compose down

dev-reset:
	cd infrastructure/compose && docker compose down -v && docker compose up --build

logs:
	cd infrastructure/compose && docker compose logs -f

# ── Testing ───────────────────────────────────────────────────────────────────

test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-e2e:
	pytest tests/e2e/ -v --tb=short

test-security:
	pytest tests/security/ -v --tb=short

# ── Code Quality ──────────────────────────────────────────────────────────────

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy packages/ services/ apps/api/ --ignore-missing-imports

# ── Database ──────────────────────────────────────────────────────────────────

migrate:
	cd apps/api && alembic upgrade head

migrate-down:
	cd apps/api && alembic downgrade -1

# ── ML Pipeline ───────────────────────────────────────────────────────────────

download-datasets:
	python ml/datasets/download.py

train-audio:
	python ml/training/train_audio.py

train-video:
	python ml/training/train_video.py

export-onnx:
	python ml/export/onnx_export.py

benchmark:
	python ml/evaluation/benchmark.py

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
