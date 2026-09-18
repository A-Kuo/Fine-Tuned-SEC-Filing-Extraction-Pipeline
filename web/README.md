# FinDoc Dashboard (web/)

Institutional-style financial disclosure and analytics dashboard, built with
Next.js 14 (App Router, TypeScript), Tailwind CSS, `lucide-react`, and
`recharts`. Replaces the earlier static `frontend/` debug console -- that
approach's constraint (no build step, so Vercel's Python functions stayed
tiny) no longer applies once the frontend itself is a Next.js app with its
own Node build.

The main page lives at `app/financial-dashboard/page.tsx`; `app/page.tsx`
redirects `/` there. Data on the page (`lib/mock-data.ts`) is illustrative,
shaped like the real records `serving/api.py` returns from
`/extractions/{filing_id}` and `/pipeline/status` -- wiring the page to that
live API is a follow-up, not done here.

## Local use

```bash
npm install
npm run dev
```

## Deploying to Vercel

Set this Vercel project's **Root Directory to `web`** (Project Settings ->
General). Next.js is auto-detected; no other build configuration is needed.

## Connecting back to the FastAPI backend

`serving/api.py`'s `GET /` 307-redirects to `config.yaml`'s
`serving.frontend_url` when set -- point that at this app's deployed URL so
visiting the API's own origin lands here instead of a bare JSON index.

The SageMaker Async Inference proxy and the read-only Supabase table browser
that lived in the old `frontend/api/*.py` Vercel functions have not been
ported to this app yet (see `sagemaker/README.md`'s Status section).
