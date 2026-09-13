# FinDocAnalyzer Debug Console

A static, framework-free debug UI for `serving/api.py`'s FastAPI backend --
built for human-interface debugging (manually poking `/extract`, `/rag/query`,
`/stats`, `/pipeline/status`), not as a production frontend.

## Why static + separate from the backend

The backend needs GPU access and heavy ML dependencies (torch, transformers,
bitsandbytes) to actually run the fine-tuned model -- fundamentally
incompatible with Vercel's serverless function constraints (size, memory,
execution time). So this frontend is a plain static site (no build step,
no framework) deployable to Vercel on its own, that talks to the FastAPI
backend wherever it's actually hosted (locally, Docker, a GPU box, etc.)
via a configurable API base URL stored in the browser's localStorage.

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

This directory is a plain static site -- Vercel auto-detects it, no
framework config needed. Point a Vercel project's root directory at
`frontend/` (or deploy this directory directly), and it just serves
`index.html`/`app.js`/`style.css` as-is.
