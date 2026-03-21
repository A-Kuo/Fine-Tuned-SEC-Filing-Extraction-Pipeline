# CLAUDE.md — Agent Context for FinDocAnalyzer

This file provides persistent context for AI coding agents working on this repository. Read it fully before making any changes.

---

## Project Summary

**FinDocAnalyzer** is a production pipeline that extracts structured financial data from SEC filings (10-K, 10-Q, 8-K) using a QLoRA fine-tuned Llama 3.1 8B model. The system:

1. Fine-tunes Llama 3.1 8B with QLoRA (NF4 4-bit + LoRA r=16) to map raw filing text → structured JSON
2. Serves predictions via FastAPI with Redis read-through cache + PostgreSQL persistent storage
3. Monitors production accuracy via statistical drift detection (two-sample proportion z-test)

**Current status:** Iteration 1 complete. All 103 tests pass. Infrastructure uses synthetic SEC data. Iteration 2 is the active next milestone (see roadmap below).

**Repository:** `https://github.com/akuo6/financial-llm`

---

## Directory Layout

```
FinDocAnalyzer/
├── CLAUDE.md                   ← this file
├── README.md                   ← human-facing documentation
├── config.yaml                 ← central config (model, training, serving, DB, monitoring)
├── .env.example                ← environment variable template (copy to .env)
├── docker-compose.yml          ← PostgreSQL 16-alpine + Redis 7-alpine
├── Makefile                    ← all common commands (run `make help`)
├── pyproject.toml              ← package metadata + ruff/mypy/pytest config
├── requirements.txt            ← Python dependencies
│
├── .github/workflows/
│   └── ci.yml                  ← GitHub Actions CI (Python 3.10/3.11/3.12 matrix)
│
├── src/                        ← core library (imported by all other modules)
│   ├── __init__.py
│   ├── config.py               ← load_config(), get_project_root(); YAML + env overrides
│   ├── model.py                ← FinancialLLM class: loads NF4 quantized base + LoRA adapters
│   ├── inference.py            ← ExtractionEngine: prompt builder + model.generate() + parser
│   ├── postprocessing.py       ← parse_extraction() 5-strategy cascade; ExtractionResult dataclass
│   └── database.py             ← RedisCache, PostgresStorage, DatabaseManager (read-through)
│
├── training/
│   ├── __init__.py
│   ├── train.py                ← QLoRA SFTTrainer; entry: python training/train.py
│   ├── callbacks.py            ← MetricsCallback (wandb-compatible), EarlyStoppingOnLoss
│   └── data_collator.py        ← FinancialDataCollator: IGNORE_INDEX=-100 on instruction tokens
│
├── serving/
│   ├── __init__.py
│   ├── api.py                  ← FastAPI app; POST /extract, POST /extract/batch, GET /health, GET /metrics
│   ├── inference_server.py     ← vLLM server wrapper (PagedAttention, continuous batching)
│   └── batch_inference.py      ← CLI: --input_dir, --server_url, --output_dir
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py             ← per-field accuracy, fuzzy_financial_match() for numeric fields
│   └── benchmark.py            ← latency (p50/p95/p99), throughput, memory profiling
│
├── monitoring/
│   ├── __init__.py
│   ├── monitor.py              ← DriftDetector: z-test; generate_full_report()
│   ├── alerts.py               ← AlertDispatcher: log / SMTP email / webhook
│   └── dashboard.py            ← Streamlit dashboard (accuracy, latency, drift history)
│
├── scripts/
│   ├── download_model.py       ← HuggingFace model download (requires HF_TOKEN in .env)
│   ├── download_dataset.py     ← synthetic SEC filing generator (creates data/*.jsonl)
│   ├── format_data.py          ← converts raw JSONL → Llama 3.1 chat template format
│   └── init_db.sql             ← PostgreSQL DDL: extractions + audit_log tables
│
├── tests/                      ← 103 tests, all run without GPU
│   ├── __init__.py
│   ├── test_postprocessing.py  ← 27 tests: JSON parsing, schema validation, edge cases
│   ├── test_database.py        ← 19 tests: cache TTL, storage, graceful degradation
│   ├── test_api.py             ← 13 tests: REST schema validation, prompt construction
│   ├── test_monitoring.py      ← 24 tests: drift detection, z-test, evaluation metrics
│   └── test_integration.py     ← 11 tests: end-to-end extraction flows (mocked model)
│
├── notebooks/                  ← Colab GPU notebooks
│   ├── train_qlora.ipynb       ← QLoRA fine-tuning (T4/L4/A100 auto-config)
│   └── inference_eval.ipynb    ← Extraction + evaluation + latency profiling
│
├── data/
│   ├── sample_10k.txt          ← reference SEC 10-K text for smoke testing
│   └── sample_10k.expected.json ← expected extraction output for sample
│
├── models/                     ← LoRA adapter checkpoints (gitignored, ~500 MB each)
└── results/                    ← evaluation/benchmark outputs (gitignored)
```

---

## Key Conventions

### Python Style
- Formatter: `ruff format` (line length 100)
- Linter: `ruff check` (E/F/I/N/W/UP rules; E501 and N802/N803 ignored)
- Type checker: `mypy --ignore-missing-imports`
- All modules use `sys.path.insert(0, str(Path(__file__).parent.parent))` at the top to ensure the project root is on `sys.path` — this is intentional for the nested package layout.
- Docstrings in all public modules follow the existing style: module-level docstring with `Usage:` example.

### Configuration
- **Never hardcode** model paths, DB credentials, or thresholds. All config comes from `config.yaml` via `load_config()`.
- Environment variables in `.env` override config.yaml values.
- Test code that needs config should call `load_config()` from `src.config`.

### Testing
- All tests must pass without a GPU. Mock `torch`, `transformers`, and model inference using `unittest.mock`.
- Database tests use the graceful-degradation path (Redis/Postgres unavailable) — do not require live connections.
- Run tests: `make test` or `python -m pytest tests/ -v`
- Do not break the 103-test baseline. Add tests for any new functionality.

### Imports
```python
# Correct pattern used throughout (modules add project root to sys.path)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.postprocessing import parse_extraction, ExtractionResult
```

---

## Common Commands

```bash
# Infrastructure
make infra-up           # docker compose up -d (Postgres + Redis)
make infra-down         # docker compose down
make db-init            # initialize schema from scripts/init_db.sql

# Data
make data               # generate + format synthetic training data

# Training
make train              # QLoRA fine-tune (requires GPU + downloaded model)

# Serving
make serve              # uvicorn serving.api:app --host 0.0.0.0 --port 8000
make serve-vllm         # vLLM inference server on port 8000

# Testing
make test               # pytest tests/ -q
make test-verbose       # pytest tests/ -v
make test-coverage      # with --cov report

# Code quality
make lint               # ruff check
make format             # ruff format
make typecheck          # mypy src/

# Monitoring
make monitor            # drift check CLI report
make dashboard          # Streamlit at localhost:8501
make evaluate           # accuracy metrics
make benchmark          # latency/throughput/memory
```

---

## Architecture Decisions (Do Not Change Without Reason)

| Decision | Rationale |
|---|---|
| QLoRA NF4 4-bit | 77% memory reduction; fits single RTX 4090 (24 GB) |
| LoRA rank 16, all 7 projection layers | Sufficient for SEC→JSON mapping; higher rank showed no gain in ablations |
| `do_sample=False` (greedy decoding) | Deterministic outputs required for audit compliance and reproducibility |
| 5-strategy JSON parser cascade | LLM outputs are unpredictable; cascade maximizes parse success rate |
| Label masking (IGNORE_INDEX=-100) on instruction tokens | Loss computed only on output JSON; prevents gradient waste on boilerplate |
| Redis + PostgreSQL two-tier storage | Redis handles repeated lookups within 1-day TTL; Postgres for compliance audit trail |
| Two-condition drift alert (threshold AND p<0.05) | Reduces false positives from normal daily variance |

---

## Extraction Schema

The model outputs JSON with these fields:

| Field | Type | Required |
|---|---|---|
| `filing_id` | string | yes |
| `company_name` | string | yes |
| `ticker` | string | no |
| `filing_type` | string (10-K/10-Q/8-K) | yes |
| `date` | string (YYYY-MM-DD) | yes |
| `fiscal_year_end` | string | no |
| `revenue` | float (millions USD) | no |
| `net_income` | float (millions USD) | no |
| `total_assets` | float (millions USD) | no |
| `total_liabilities` | float (millions USD) | no |
| `eps` | float | no |
| `sector` | string | no |

Validation is in `src/postprocessing.py → validate_extraction()`. The confidence threshold (default 0.7) is in `config.yaml → extraction.confidence_threshold`.

---

## GPU Acceleration / Colab

The GPU-dependent components are:
- **Training** (`training/train.py`) — requires CUDA for NF4 quantization + LoRA fine-tuning
- **Model loading** (`src/model.py`) — `device_map="auto"` places layers on GPU; `BitsAndBytesConfig` requires CUDA
- **Inference** (`src/model.py → generate()`) — moves input tensors to CUDA if available
- **vLLM serving** (`serving/inference_server.py`) — requires Linux + CUDA (no Windows/MPS support)

All GPU code uses `torch.cuda.is_available()` guards. Tests mock out all GPU paths.

**Colab notebooks** in `notebooks/` are the primary way to run GPU workloads without local hardware:
- `train_qlora.ipynb` — auto-configures batch size by GPU tier (T4=4, L4=6, A100=8), saves adapter to Google Drive
- `inference_eval.ipynb` — loads adapter from Drive, runs extraction + evaluation + latency profiling

Batch size configuration by GPU tier:

| GPU | VRAM | Batch | Grad Accum | Effective Batch |
|-----|------|-------|------------|-----------------|
| T4  | 16 GB | 4 | 8 | 32 |
| L4/A10 | 24 GB | 6 | 6 | 36 |
| A100 | 40 GB | 8 | 4 | 32 |

When editing notebooks, preserve the cell ordering and markdown section headers — they serve as the user's guide through the pipeline.

---

## Active Iteration: Iteration 2

**Goal:** Replace synthetic training data with real SEC filings and improve accuracy tracking.

### Task List for Iteration 2

- [ ] **EDGAR integration** — Add `scripts/fetch_edgar.py` using the [EDGAR full-text search API](https://efts.sec.gov/LATEST/search-index?q=%2210-K%22&dateRange=custom&startdt=2023-01-01&enddt=2024-12-31&forms=10-K). Key endpoint: `https://data.sec.gov/submissions/CIK{cik:010d}.json`. Rate limit: 10 req/sec. Use `User-Agent: FinDocAnalyzer contact@example.com` header (SEC requirement). Parse XBRL inline data for ground-truth numeric fields where available.
- [ ] **Hyperparameter sweep** — Grid search over LoRA rank (8/16/32), learning rate (1e-4/5e-4/1e-3), and target modules (attention-only vs attention+FFN). Log to W&B or MLflow. Store sweep results in `results/sweep/`.
- [ ] **A/B testing framework** — `src/ab_router.py`: route extraction requests across two model versions (configurable split ratio), log per-version accuracy to PostgreSQL, expose `/ab/report` endpoint in `serving/api.py`.
- [ ] **Grafana + Prometheus** — Add Prometheus scrape endpoint (already have `prometheus-client` in requirements). Add `docker-compose.yml` services for Prometheus + Grafana. Import pre-built Grafana dashboard for latency/accuracy panels.

### Iteration 2 Notes
- Ground truth for EDGAR data: parse XBRL `us-gaap:Revenues`, `us-gaap:NetIncomeLoss`, `us-gaap:Assets` etc. These are the authoritative values; use them to build labeled evaluation sets.
- The `fuzzy_financial_match()` function in `evaluation/evaluate.py` accepts a tolerance parameter — keep this at ±5% for numeric fields (standard for reported vs extracted variance due to units/rounding).
- When fetching real filings, strip boilerplate headers/footers before sending to model. The context window is 2048 tokens; most 10-K Item 7 (MD&A) sections exceed this — implement chunking with overlap in `scripts/fetch_edgar.py`.

---

## Infrastructure

```yaml
# docker-compose.yml services
postgres:
  image: postgres:16-alpine
  container_name: financial-llm-postgres
  ports: 5432
  database: financial_llm
  user: finllm
  password: finllm_dev  # override via POSTGRES_PASSWORD in .env

redis:
  image: redis:7-alpine
  container_name: financial-llm-redis
  ports: 6379
  max_memory: 256mb (LRU eviction)
```

Database schema is in `scripts/init_db.sql`. Two tables:
- `extractions` — one row per extraction (filing_id, all fields, confidence, latency_ms, created_at)
- `audit_log` — immutable append-only log for compliance

---

## Environment Variables

Required in `.env` (copy from `.env.example`):

```bash
HF_TOKEN=             # HuggingFace token for model download
POSTGRES_PASSWORD=    # override default finllm_dev for non-dev environments
ALERT_EMAIL=          # optional: email address for drift alerts
```

---

## CI

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on push/PR to `main`:
- **test job**: Python 3.10/3.11/3.12 matrix; generates synthetic data, runs `pytest tests/`
- **lint job**: Python 3.12; runs `ruff check` with `--exit-zero` (non-blocking)

GPU-dependent packages (`torch`, `transformers`, `bitsandbytes`, `peft`, `trl`, `vllm`) are not installed in CI. All tests mock these out.

---

## Known Issues / Technical Debt

1. **Flat-to-nested migration** — Files were originally committed flat at the project root. They were reorganized into subdirectories. If any import fails with `ModuleNotFoundError`, verify `sys.path.insert(0, project_root)` is present at the top of the offending file.
2. **Synthetic data only** — Training data is generated by `scripts/download_dataset.py` using a template-based synthetic generator. Model accuracy numbers (94%) are measured against this synthetic test set — real-world accuracy on EDGAR filings is unknown until Iteration 2.
3. **`data/*.txt` gitignore exception** — The `.gitignore` ignores all `data/*.txt` but explicitly un-ignores `data/sample_10k.txt`. Confirm this is tracked: `git ls-files data/`.
4. **vLLM on Windows** — `vllm` does not officially support Windows. The vLLM serving path requires Linux/WSL2. The FastAPI standalone mode works cross-platform.
5. **`tests/__init__.py`** — Added to enable correct relative imports. If pytest has discovery issues, check `pyproject.toml → tool.pytest.ini_options.testpaths`.
