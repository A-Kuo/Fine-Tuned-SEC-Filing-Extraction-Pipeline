# FinDocAnalyzer Debug Console

A static, framework-free debug UI -- built for human-interface debugging
(manually poking `/extract`, `/rag/query`, `/stats`, `/pipeline/status`),
not as a production frontend. Supports two backend modes, toggled in the
UI's connection panel:

- **Direct FastAPI** -- talks straight to `serving/api.py` wherever it's
  hosted (local, Docker, a GPU box) via a configurable API base URL.
- **SageMaker via proxy** -- talks to this same origin's `api/` (below),
  which forwards to a real SageMaker Async Inference endpoint. See
  `sagemaker/README.md` for that architecture and its current status (as
  of writing: code exists, no live endpoint does yet).

## Why static + separate from the model-serving backend

The backend needs GPU access and heavy ML dependencies (torch, transformers,
bitsandbytes) to actually run the fine-tuned model -- fundamentally
incompatible with Vercel's serverless function constraints (size, memory,
execution time). So this frontend is a plain static site (no build step,
no framework) deployable to Vercel on its own. `api/` is the one exception:
a genuinely tiny proxy (boto3 only) that Vercel's Python runtime handles
fine -- see `sagemaker/README.md` for why a proxy is required at all
(SageMaker's InvokeEndpoint needs AWS SigV4 signing, not a plain browser fetch()).

## Local use

Open `index.html` directly in a browser, or serve the directory:

```bash
python -m http.server 3000 --directory frontend
```

Then set the API base URL in the connection panel to wherever
`serving/api.py` is actually running (default assumption: `http://localhost:8000`).

## CORS

The backend must allow this frontend's origin. `config.yaml`'s
`security.cors_origins` defaults to `["*"]`, which already covers this --
no backend change needed for a default setup. Tighten it to the frontend's
actual deployed origin if `cors_origins` is ever locked down.

## Deploying to Vercel

Point a Vercel project's Root Directory at `frontend/` (or deploy this
directory directly). Vercel auto-detects the static site (`index.html`/
`app.js`/`style.css`) and the `api/` proxy functions (`requirements.txt`
in this directory covers their one dependency, boto3) with no framework
config needed.

For SageMaker mode to actually work, set these on the Vercel project
(dashboard -> Settings -> Environment Variables -- never commit these):
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`SAGEMAKER_ENDPOINT_NAME`, `SAGEMAKER_S3_BUCKET`. Without them, `api/invoke.py`
returns a clear 503 rather than a confusing failure.
