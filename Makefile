.PHONY: test test-verbose lint format data train serve serve-vllm serve-pipeline \
        monitor benchmark evaluate dashboard \
        infra-up infra-down db-init \
        pipeline-up pipeline-down pipeline-status pipeline-logs \
        pipeline-extract pipeline-enrich pipeline-visualize pipeline-run \
        clean clean-all help

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

fetch-edgar:  ## Fetch real SEC filings (respects SEC rate limits; set EDGAR tickers in config)
	python scripts/fetch_edgar.py

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

# ─── Pipeline Integration ────────────────────────────────────────────────────

pipeline-up:  ## Start full pipeline stack (FinDoc + Ticker + Viz)
	docker compose -f docker-compose.pipeline.yml up -d
	@echo "Pipeline starting..."
	@echo "FinDocAnalyzer: http://localhost:8000"
	@echo "TickerAgent:    http://localhost:8002"
	@echo "VizFramework:   http://localhost:8003"
	@echo "Streamlit:      http://localhost:8502"

pipeline-down:  ## Stop full pipeline stack
	docker compose -f docker-compose.pipeline.yml down

pipeline-status:  ## Check pipeline service status
	@docker compose -f docker-compose.pipeline.yml ps

pipeline-logs:  ## View pipeline logs
	docker compose -f docker-compose.pipeline.yml logs -f

pipeline-extract:  ## Run extraction stage only
	@echo "Running extraction on sample data..."
	python serving/batch_inference.py \
		--input_dir data/filings/ \
		--server_url http://localhost:8000 \
		--webhook_url http://localhost:8002/ingest

pipeline-enrich:  ## Run enrichment stage only (requires extracted data)
	@echo "Enriching tickers from database..."
	curl -X POST http://localhost:8002/batch-enrich \
		-H "Content-Type: application/json" \
		-d '{"limit": 10, "min_confidence": 0.8}'

pipeline-visualize:  ## Run visualization stage only (requires enriched data)
	@echo "Generating dashboards..."
	curl -X POST http://localhost:8003/render-batch \
		-H "Content-Type: application/json" \
		-d '{"chart_type": "financial_summary", "limit": 10}'

pipeline-run:  ## Run full pipeline end-to-end
	@echo "=== Pipeline Stage 1: Extraction ==="
	$(MAKE) pipeline-extract
	@echo "Waiting for extraction to complete..."
	@sleep 5
	@echo "=== Pipeline Stage 2: Enrichment ==="
	$(MAKE) pipeline-enrich
	@echo "Waiting for enrichment to complete..."
	@sleep 5
	@echo "=== Pipeline Stage 3: Visualization ==="
	$(MAKE) pipeline-visualize
	@echo "Pipeline complete! View dashboards at http://localhost:8502"

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
