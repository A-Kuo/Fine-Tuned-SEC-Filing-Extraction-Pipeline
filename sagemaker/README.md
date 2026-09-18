# SageMaker Async Inference deployment

Scaffolding for deploying the fine-tuned QLoRA adapter to a real GPU
endpoint, reachable from the Vercel frontend (`web/`, a Next.js app --
replaced the earlier static `frontend/` site) through a thin proxy.
**Written but not deployed** -- no AWS resources exist from this directory
as of this commit, and the proxy route handlers themselves still need to
be ported to `web/app/api/` (TypeScript) now that the frontend is Next.js
rather than static HTML with Python serverless functions. See "Status"
below for exactly what's real versus what's still needed.

## Why this architecture (and why not simpler)

- **Why not deploy `serving/api.py` itself to Vercel?** It needs torch/
  transformers/bitsandbytes/vLLM (well over Vercel's Python function size
  limit) and a GPU (Vercel has none). Tried and reverted -- see the git
  history around `pyproject.toml`'s now-removed `[tool.vercel]` entrypoint.
- **Why not call SageMaker directly from the browser?** `InvokeEndpoint`
  requires AWS Signature Version 4 request signing. A browser cannot hold
  AWS credentials safely to sign with client-side. A server-side proxy
  (route handlers under `web/app/api/`, Vercel serverless functions) is
  genuinely required, not a nice-to-have.
- **Why Asynchronous Inference specifically, not Serverless Inference or a
  real-time endpoint?**
  | Mode | GPU | Scales to zero | Fits this model? |
  |---|---|---|---|
  | Serverless Inference | No (CPU only, 6GB max) | Yes | No -- needs ~7.2GB resident, GPU |
  | Real-time endpoint | Yes | No (bills 24/7) | Works, but expensive for sporadic demo traffic |
  | **Async Inference** | Yes | Yes (via Application Auto Scaling) | Best fit |

  This repo tried and removed a SageMaker deployment once before (commit
  `ef518cb`, "Migrate training pipeline from AWS SageMaker to MLFlow +
  Kaggle") -- the reason wasn't confirmed when this was rebuilt, so this
  is a considered retry (Async Inference specifically addresses the
  continuous-billing concern a real-time endpoint would have), not an
  assumption that the prior removal's reasoning no longer applies.

## Files

- **`inference.py`** -- the SageMaker inference-toolkit contract
  (`model_fn`/`input_fn`/`predict_fn`/`output_fn`). Reuses this repo's own
  `FinancialLLM` (`src/extraction/model.py`) for model loading -- local
  and SageMaker serving load the model identically. One endpoint serves
  both `/extract`-shaped and `/rag/query`-shaped requests via a `"task"`
  field in the input JSON (RAG retrieval itself happens outside the
  endpoint, before invocation -- this container has no Postgres access).
- **`requirements.txt`** -- the inference container's dependencies (a
  subset of the root `requirements.txt`: no mlflow/fastapi/dagshub, just
  what's needed to load and run the model).
- **`deploy_async_endpoint.py`** -- packages the LoRA adapter (small,
  ~85MB, NOT the base model weights) into `model.tar.gz`, uploads it to
  S3, and creates the Model/EndpointConfig/Endpoint plus a scale-to-zero
  Application Auto Scaling policy. **Defaults to a dry run** (prints what
  it would do, touches no AWS) -- pass `--yes-really-deploy` for the real
  thing, which costs money and needs a real trained adapter to package.

## Status: what's real vs. what's still needed

**Real (exists now):**
- All code in this directory, syntax-checked and (for `inference.py`'s
  pure request-handling logic) unit-tested (`tests/test_sagemaker_inference.py`).
- The architectural decision and its documented rationale above.

**Not yet real:**
- **No trained adapter exists.** `models/` doesn't exist in this repo;
  `.github/workflows/kaggle_training.yml` (fixed earlier this session to
  actually generate its training data) has never been run. Run that
  workflow and download its artifact before attempting a real deploy --
  `deploy_async_endpoint.py` refuses to package a missing/incomplete
  adapter directory rather than silently deploying garbage.
- **No AWS resources exist.** No S3 bucket, IAM role, SageMaker Model/
  EndpointConfig/Endpoint, or Auto Scaling policy has been created.
- **`deploy_async_endpoint.py`'s container image is a placeholder**
  (`REPLACE_WITH_REAL_ECR_IMAGE_URI`) -- pick a real SageMaker-provided
  PyTorch/HuggingFace inference DLC URI for your region before a real run.
- **The Vercel proxy doesn't exist yet in `web/`** -- it still needs to be
  ported from the old Python `frontend/api/invoke.py`/`status.py` into
  TypeScript route handlers, and its `SAGEMAKER_ENDPOINT_NAME`/
  `SAGEMAKER_S3_BUCKET` environment variables aren't set on any Vercel
  project.

## Deploying for real (once a trained adapter exists)

```bash
python sagemaker/deploy_async_endpoint.py \
    --adapter-dir models/llama-sec-v1 \
    --role-arn arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole \
    --s3-bucket your-bucket-name \
    --yes-really-deploy
```

Then set on the Vercel project (dashboard -> Settings -> Environment
Variables, never committed to git): `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `SAGEMAKER_ENDPOINT_NAME`,
`SAGEMAKER_S3_BUCKET`. Switch the debug console's Mode dropdown to
"SageMaker via proxy."
