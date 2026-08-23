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
