# AGENTS.md

## Cursor Cloud specific instructions

### Environment

- **Update script** (runs on pod startup): installs `requirements-ci.txt`, regenerates synthetic training data. See `.cursor/environment.json` for full `install` / `terminals`.
- **Secrets**: Add `HF_TOKEN` (and optional `POSTGRES_PASSWORD`, `API_KEYS`) via Cursor **Secrets**, not committed files. `.env` is gitignored; load locally with `set -a && source .env && set +a`.
- **Python**: use `python3` and `pip`; CI deps are in `requirements-ci.txt` (no torch/GPU). Full ML stack: `pip install -r requirements.txt` (needs GPU).

### Services

| Service | Required for tests? | Start |
| --- | --- | --- |
| FastAPI API | No (pytest uses TestClient) | `make serve` or terminal `api` from environment.json |
| PostgreSQL + Redis | No (graceful degradation) | `make infra-up` (requires Docker) |

### Verify

```bash
make test          # 129 tests, no GPU/services
make lint          # needs: pip install ruff
curl localhost:8000/health   # when API terminal is running
```

### Gotchas

- `/extract` lazy-loads the model and needs `torch`, GPU, and downloaded weights (`HF_TOKEN` + `scripts/download_model.py`). `/health` works without a model.
- `make db-init` uses user `postgres` but compose creates `finllm` — schema is auto-applied via `docker-entrypoint-initdb.d` on first `infra-up`.
- Do not commit `.env` or force-add secrets; CI blocks tracked `.env*` files except `.env.example`.

### Training (Kaggle vs local)

- **Do not run `make train-kaggle` from Cloud Agents** unless the user explicitly wants to overwrite the remote kernel. `scripts/submit_kaggle_job.py` runs `kaggle kernels push`, which **replaces** whatever is currently in the Kaggle kernel (`augustinekuo/findoc-qlora-train`) with `scripts/kaggle_kernel/train_kernel.py` + `kernel-metadata.json` from this repo.
- **Source of truth:** If the user has custom logic in the Kaggle editor (e.g. `/edit/run/...`), pull that into `scripts/kaggle_kernel/train_kernel.py` *before* the next push, or confirm the repo file is authoritative.
- **Kaggle Notebook secrets** (Add-ons → Secrets, not files): `DAGSHUB_USER_TOKEN`, `HF_TOKEN`. The repo kernel loads these via `kaggle_secrets.UserSecretsClient`.
- **Local GPU path:** `make train` on a machine with CUDA + `.env` (repo root). Prefer the user's own machine for training; this Cloud VM is typically CPU-only and has no Docker.
- Repo kernel behavior: clones this GitHub repo, `pip install -r requirements.txt`, runs `training/train.py` (same path as local; logs to DagsHub/MLFlow).
