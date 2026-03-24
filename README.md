# FinDocAnalyzer

**SEC Filing Extraction Pipeline — QLoRA Fine-Tuned Llama 3.1 8B**

[![CI](https://github.com/akuo6/financial-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/akuo6/financial-llm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production system for extracting structured financial data from SEC 10-K/10-Q/8-K filings using a domain-adapted LLM. Fine-tunes Llama 3.1 8B with QLoRA (4-bit quantization + LoRA adapters), serving predictions through a FastAPI REST layer backed by Redis + PostgreSQL.

---

## Performance

| Metric | Value | Baseline |
|---|---|---|
| Extraction Accuracy | 94% (fully correct JSON outputs) | |
| Field-Level Accuracy | 92–99% per field | |
| Inference Latency (p50) | ~320 ms/doc | |
| Throughput | ~60 docs/min | |
| Memory Footprint | 7.2 GB (NF4 4-bit) | 32 GB for same Llama 8B model at FP32 (77% reduction) |
| Trainable Parameters | ~200M / 8B total (2.5%) | |
| Cost per Document | ~$0.003 self-hosted | ~$0.50 via GPT-4 API (167x reduction) |

---

## Architecture

### Single-Component View

```
SEC Filing Text
      │
      ▼
┌─────────────────┐
│  Prompt Builder  │  Llama 3.1 chat template (system + extraction instruction)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Llama 3.1 8B   │  NF4 4-bit frozen weights
│  + LoRA Adapters│  r=16, α=32 — q/k/v/o/gate/up/down_proj
└────────┬────────┘
         ▼
┌─────────────────┐
│  JSON Parser     │  5-strategy cascade (direct → strip fences →
│  + Validator     │  regex extract → fix truncation → field-level fallback)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Redis Cache     │  1ms reads, 1-day TTL (LRU, 256 MB)
│  + PostgreSQL    │  Persistent storage + audit trail
└────────┬────────┘
         ▼
┌─────────────────┐
│  Monitoring      │  Drift detection (z-test), latency SLAs, Streamlit dashboard
└─────────────────┘
```

### Full Pipeline Integration

FinDocAnalyzer is the **extraction layer** in a three-stage financial intelligence pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINANCIAL INTELLIGENCE PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1: EXTRACTION (This Repo)          Stage 2: ENRICHMENT                     Stage 3: VISUALIZATION
┌─────────────────────┐                 ┌─────────────────────┐                 ┌─────────────────────┐
│  SEC EDGAR Filing   │                 │  Ticker Symbol      │                 │  Combined Data      │
│  (10-K/10-Q/8-K)   │───────────────▶│  + Market Data      │───────────────▶│  Dashboard          │
└─────────┬───────────┘                 └─────────────────────┘                 └─────────────────────┘
          │                                      ▲                                       ▲
          ▼                                      │                                       │
┌─────────────────────┐                          │                                       │
│  FinDocAnalyzer     │  Extracted ticker ───────┘                                       │
│  • Company name     │  + financials                                                    │
│  • Revenue          │                                                                   │
│  • Net income       │  ────────────────────────────────────────────────────────────────┘
│  • Ticker symbol    │         (via shared PostgreSQL or webhook)
└─────────────────────┘

Data Flow:
1. SEC Filing → Structured financials (JSON) + Ticker symbol
2. Ticker symbol → Real-time market intelligence (price, trends, news sentiment)
3. Combined → Interactive dashboards with drill-down capability
```

| Repository | Role | Consumes | Produces |
|------------|------|----------|----------|
| **[FinDocAnalyzer](https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline)** (this repo) | **Document Extraction Layer** | Raw SEC filings | Structured financial data + ticker symbols |
| **[Ticker-Analyzer-Agent](https://github.com/A-Kuo/Financial-Economic-Ticker-Analyzer-Agent)** | **Market Intelligence Layer** | Ticker symbols + extracted financials | Real-time market context, trends, sentiment |
| **[Agentic-Viz-Framework](https://github.com/A-Kuo/Agentic-Visualization-Framework)** | **Presentation Layer** | Any structured data | Interactive dashboards, charts, drill-downs |

**Integration Modes:**
- **Database-linked**: FinDocAnalyzer writes to PostgreSQL; Ticker Analyzer polls for new tickers
- **Webhook**: FinDocAnalyzer POSTs extracted ticker to Ticker Analyzer's `/ingest` endpoint
- **Batch pipeline**: `make pipeline-run` orchestrates all three stages sequentially

**Extracted fields:** `filing_id`, `company_name`, `ticker`, `filing_type`, `date`, `fiscal_year_end`, `revenue`, `net_income`, `total_assets`, `total_liabilities`, `eps`, `sector`

---

## Project Structure

```
FinDocAnalyzer/
├── config.yaml                 # Central configuration (model, training, serving, DB)
├── docker-compose.yml          # PostgreSQL 16 + Redis 7
├── Makefile                    # Dev/ops commands
├── pyproject.toml              # Package metadata + ruff/mypy/pytest config
├── requirements.txt            # Python dependencies
│
├── src/                        # Core library
│   ├── config.py               # YAML config loader with env overrides
│   ├── model.py                # FinancialLLM: quantized base + LoRA adapter loading
│   ├── inference.py            # ExtractionEngine: prompt → model → structured result
│   ├── postprocessing.py       # 5-strategy JSON parser + schema validation
│   └── database.py             # RedisCache + PostgresStorage + DatabaseManager
│
├── training/                   # Fine-tuning pipeline
│   ├── train.py                # QLoRA training with SFTTrainer
│   ├── callbacks.py            # Metrics logging + early stopping
│   └── data_collator.py        # Label masking (loss computed only on output tokens)
│
├── serving/                    # Production serving layer
│   ├── api.py                  # FastAPI endpoints: /extract, /extract/batch, /health, /metrics
│   ├── inference_server.py     # vLLM server wrapper (PagedAttention, continuous batching)
│   └── batch_inference.py      # CLI for bulk processing from a directory of filings
│
├── evaluation/                 # Accuracy and performance measurement
│   ├── evaluate.py             # Per-field accuracy, fuzzy numeric matching
│   └── benchmark.py            # Latency, throughput, memory profiling
│
├── monitoring/                 # Production observability
│   ├── monitor.py              # Drift detection via two-sample proportion z-test
│   ├── alerts.py               # Alert dispatch (log / email / webhook)
│   └── dashboard.py            # Streamlit dashboard
│
├── scripts/                    # Data preparation and model setup
│   ├── download_model.py       # Fetch Llama 3.1 from HuggingFace Hub
│   ├── download_dataset.py     # Generate synthetic SEC filing training pairs
│   ├── format_data.py          # Convert to Llama 3.1 chat template format
│   └── init_db.sql             # PostgreSQL schema (extractions + audit log tables)
│
├── tests/                      # 103 tests, no GPU required
│   ├── test_postprocessing.py  # JSON parsing and validation (27 tests)
│   ├── test_database.py        # Cache, storage, graceful degradation (19 tests)
│   ├── test_api.py             # REST schemas, prompt building (13 tests)
│   ├── test_monitoring.py      # Drift detection, evaluation metrics (24 tests)
│   └── test_integration.py     # End-to-end pipeline flows (11 tests) [no GPU]
│
├── notebooks/                  # Colab notebooks (GPU-accelerated)
│   ├── train_qlora.ipynb       # QLoRA fine-tuning on Colab T4/A100
│   └── inference_eval.ipynb    # Extraction, evaluation, latency profiling
│
├── data/                       # Training data (generated) + reference samples
│   ├── sample_10k.txt          # Reference SEC 10-K text for smoke testing
│   └── sample_10k.expected.json # Expected extraction output for the sample
│
├── models/                     # Saved LoRA adapters (gitignored, see .gitignore)
└── results/                    # Evaluation outputs (gitignored)
```

---

## GPU / Colab

If you don't have a local CUDA GPU, use the provided Colab notebooks:

| Notebook | Purpose | Min GPU |
|---|---|---|
| [`notebooks/train_qlora.ipynb`](notebooks/train_qlora.ipynb) | QLoRA fine-tuning | T4 (16 GB) |
| [`notebooks/inference_eval.ipynb`](notebooks/inference_eval.ipynb) | Extraction + evaluation + latency profiling | T4 (16 GB) |

Both notebooks auto-detect GPU tier (T4/L4/A100), configure batch sizes accordingly, and save artifacts to Google Drive for persistence. Open in Colab via **Runtime → Change runtime type → GPU**.

---

## Prerequisites

- Python 3.10+
- CUDA GPU with ≥16 GB VRAM (training and inference; tests run CPU-only) — or use Colab
- Docker (for PostgreSQL + Redis)
- HuggingFace account with access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

---

## Setup

```bash
git clone https://github.com/akuo6/financial-llm.git
cd financial-llm

pip install -r requirements.txt

cp .env.example .env
# Add your HuggingFace token to .env

make infra-up       # Start PostgreSQL + Redis
make db-init        # Initialize schema
make data           # Generate synthetic training data
make test           # Run 103 tests (no GPU needed)
```

> **Security Note:** The default `POSTGRES_PASSWORD=finllm_dev` in `docker-compose.yml` is for local development only. Override with the `POSTGRES_PASSWORD` environment variable for any non-local deployment.

---

## Training

```bash
python scripts/download_model.py     # ~16 GB download, requires HF_TOKEN

make train
# Equivalent: python training/train.py --num_epochs 3 --batch_size 8 --learning_rate 5e-4
# Output: models/llama-sec-v1/  (LoRA adapters only, ~500 MB)
```

Key training config (see `config.yaml`):

| Parameter | Value | Rationale |
|---|---|---|
| LoRA rank | 16 | Sufficient for domain-specific extraction task |
| LoRA alpha | 32 | 2× rank — standard scaling |
| Target modules | q/k/v/o/gate/up/down | All linear projections |
| Batch size | 8 × 4 grad accum = 32 effective | |
| Learning rate | 5e-4 with cosine decay | Standard for LoRA |
| Quantization | NF4 4-bit + double quant | 7.2 GB vs 32 GB FP32 |

---

## Inference

```bash
# Start API server (standalone, loads model directly)
make serve

# Single extraction
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "SEC FILING - FORM 10-K\nRegistrant: Apple Inc..."}'

# Batch processing from a directory
python serving/batch_inference.py --input_dir data/filings/ --server_url http://localhost:8000

# Production: use vLLM backend for higher throughput
make serve-vllm &
uvicorn serving.api:app --host 0.0.0.0 --port 8001
```

For production throughput, the vLLM backend (PagedAttention + continuous batching) handles raw inference while the FastAPI layer handles prompt construction, post-processing, caching, and monitoring.

---

## Monitoring

```bash
make dashboard      # Streamlit dashboard at http://localhost:8501
make monitor        # CLI drift report
make evaluate       # Accuracy metrics against ground truth
make benchmark      # Latency/throughput/memory profile
```

Drift detection fires when accuracy drops below `0.92` threshold **and** the drop is statistically significant (two-sample proportion z-test, p < 0.05). With 50 daily samples, this detects a 5% absolute accuracy drop within one day at 80% statistical power.

---

## Development

```bash
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy
make test-coverage  # pytest with coverage report
```

All tests run without a GPU. The test suite mocks model inference and uses an in-memory SQLite substitute for the database layer, so CI runs fast on standard runners.

---

## Configuration

All runtime configuration lives in `config.yaml`, with environment variable overrides via `.env`. Key sections:

- **`model`** — base model path, adapter path, sequence length
- **`quantization`** — NF4 4-bit settings
- **`lora`** — rank, alpha, target modules
- **`training`** — epochs, batch size, learning rate schedule
- **`serving`** — host, port, batch size, timeouts
- **`database`** — PostgreSQL + Redis connection params
- **`monitoring`** — accuracy threshold, latency SLAs, alert config
- **`extraction`** — required/optional field definitions, confidence threshold

---

## Pipeline Integration

FinDocAnalyzer can run standalone or as part of the integrated financial intelligence pipeline.

### Quick Start: Full Pipeline

```bash
# Clone all three repositories
git clone https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline.git findoc-analyzer
git clone https://github.com/A-Kuo/Financial-Economic-Ticker-Analyzer-Agent.git ticker-agent
git clone https://github.com/A-Kuo/Agentic-Visualization-Framework.git viz-framework

# Start the full stack
cd findoc-analyzer
docker compose -f docker-compose.pipeline.yml up -d

# Run end-to-end pipeline
make pipeline-extract    # Extract from SEC filings
make pipeline-enrich     # Enrich with market data
make pipeline-visualize  # Generate dashboard
```

### Pipeline Data Schema

**Inter-service contract** (JSON Schema in `schemas/pipeline-v1.json`):

```json
{
  "extraction_id": "uuid",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "financials": {
    "revenue": 394_328_000_000,
    "net_income": 99_803_000_000,
    "filing_date": "2024-09-28"
  },
  "market_context": {        // Populated by Ticker Analyzer
    "current_price": 189.52,
    "52_week_high": 199.62,
    "sentiment_score": 0.72
  },
  "viz_config": {             // Used by Viz Framework
    "chart_type": "financial_summary",
    "drill_down": true
  }
}
```

### API Endpoints for Downstream Integration

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/extract` | POST | Single filing extraction |
| `/extract/batch` | POST | Batch extraction with callback URL |
| `/webhook/register` | POST | Register downstream service URL |
| `/pipeline/status` | GET | Current pipeline stage status |

---

## Roadmap

**Iteration 2**
- Real SEC filings via EDGAR full-text search API (replacing synthetic data)
- Hyperparameter sweep: LoRA rank, learning rate, target module ablation
- A/B testing framework for adapter version comparison
- Prometheus metrics export + Grafana dashboards
- **Pipeline integration**: Webhook callbacks to Ticker Analyzer

**Iteration 3**
- Multi-GPU training with DeepSpeed ZeRO-3
- Streaming inference for long documents (>2048 tokens)
- RLHF alignment using analyst-reviewed extraction pairs
- Kubernetes deployment manifests
- **Pipeline integration**: Event-driven architecture with message queue

---

## License

MIT — see [LICENSE](LICENSE).
