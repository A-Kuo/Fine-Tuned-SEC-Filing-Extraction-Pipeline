.PHONY: test test-verbose lint format data train serve monitor benchmark clean help

# ─── Testing ─────────────────────────────────────────────────────────────────

test:  ## Run all tests
	python -m pytest tests/ -q

test-verbose:  ## Run tests with full output
	python -m pytest tests/ -v

test-coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=src --cov=serving --cov=monitoring --cov=evaluation --cov-report=term-missing

# ─── Code Quality ────────────────────────────────────────────────────────────

lint:  ## Run linter
	python -m ruff check src/ serving/ monitoring/ evaluation/ tests/

format:  ## Auto-format code
	python -m ruff format src/ serving/ monitoring/ evaluation/ tests/

typecheck:  ## Run type checker
	python -m mypy src/ --ignore-missing-imports

# ─── Data Pipeline ───────────────────────────────────────────────────────────

data:  ## Generate training data
	python scripts/download_dataset.py
	python scripts/format_data.py

# ─── Training ────────────────────────────────────────────────────────────────

train:  ## Fine-tune model with QLoRA
	python training/train.py --num_epochs 3 --batch_size 8 --learning_rate 5e-4

# ─── Serving ─────────────────────────────────────────────────────────────────

serve:  ## Start FastAPI server
	uvicorn serving.api:app --host 0.0.0.0 --port 8000

serve-vllm:  ## Start vLLM inference server
	python serving/inference_server.py --port 8000

# ─── Infrastructure ──────────────────────────────────────────────────────────

infra-up:  ## Start PostgreSQL + Redis
	docker compose up -d

infra-down:  ## Stop infrastructure
	docker compose down

db-init:  ## Initialize database schema
	docker exec -i financial-llm-postgres psql -U postgres -d financial_llm < scripts/init_db.sql

# ─── Monitoring & Evaluation ─────────────────────────────────────────────────

monitor:  ## Run monitoring check
	python monitoring/monitor.py --full-report

dashboard:  ## Start Streamlit monitoring dashboard
	streamlit run monitoring/dashboard.py

benchmark:  ## Run performance benchmark
	python evaluation/benchmark.py --simulate --output results/benchmark.json

evaluate:  ## Evaluate model accuracy
	python evaluation/evaluate.py --generate-sample-metrics --output results/metrics.json

# ─── Housekeeping ────────────────────────────────────────────────────────────

clean:  ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache .mypy_cache
	rm -rf results/*.json results/*.jsonl

clean-all: clean  ## Remove everything including data and models
	rm -rf data/*.jsonl data/*.txt data/*.json
	rm -rf models/llama-sec-v1/

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
