// FinDocAnalyzer Debug Console -- vanilla JS, no build step, no framework.
//
// Two backend modes:
//   "direct"    -- talks straight to a FastAPI backend (serving/api.py)
//                  via the API base URL below. For local/Docker use.
//   "sagemaker" -- talks to this same origin's frontend/api/ proxy, which
//                  forwards to a SageMaker Async Inference endpoint
//                  (invoke.py) and polls for the result (status.py). See
//                  sagemaker/README.md for why a proxy is required at all
//                  (SigV4 signing, not something a browser can do).
//
// Neither mode deploys the model-serving backend itself on Vercel -- it
// needs GPU/heavy ML deps this frontend's static hosting can't provide.

const API_BASE_KEY = "findoc_debug_api_base";
const MODE_KEY = "findoc_debug_backend_mode";
const DEFAULT_API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 3000;
const POLL_TIMEOUT_MS = 5 * 60 * 1000; // cold-started GPU instances can take minutes

function getApiBase() {
  return localStorage.getItem(API_BASE_KEY) || DEFAULT_API_BASE;
}

function setApiBase(url) {
  localStorage.setItem(API_BASE_KEY, url.replace(/\/+$/, ""));
}

function getMode() {
  return localStorage.getItem(MODE_KEY) || "direct";
}

function setMode(mode) {
  localStorage.setItem(MODE_KEY, mode);
}

function showOutput(el, data, isError) {
  el.hidden = false;
  el.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
  el.classList.remove("ok", "err");
  el.classList.add(isError ? "err" : "ok");
}

async function apiFetch(path, options = {}) {
  const base = getApiBase();
  const url = `${base}${path}`;
  try {
    const resp = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    });
    let body;
    const text = await resp.text();
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
    return { ok: resp.ok, status: resp.status, body };
  } catch (e) {
    return { ok: false, status: 0, body: `Network error: ${e.message} (is the API reachable at ${base}, and does its CORS config allow this origin?)` };
  }
}

// --- SageMaker proxy (invoke + poll) ---

async function sagemakerInvoke(task, extraFields) {
  const resp = await fetch("/api/invoke", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task, ...extraFields }),
  });
  const body = await resp.json();
  if (!resp.ok) {
    return { ok: false, status: resp.status, body };
  }
  return pollSagemakerResult(body.output_location);
}

async function pollSagemakerResult(outputLocation) {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const resp = await fetch(`/api/status?output_location=${encodeURIComponent(outputLocation)}`);
    const body = await resp.json();
    if (!resp.ok) return { ok: false, status: resp.status, body };
    if (body.status === "complete") return { ok: true, status: 200, body: body.result };
    if (body.status === "failed") return { ok: false, status: 500, body };
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
  }
  return { ok: false, status: 0, body: { error: `Timed out after ${POLL_TIMEOUT_MS / 1000}s waiting for the SageMaker endpoint (cold start can take a few minutes -- try again, it may now be warm)` } };
}

// --- Connection panel ---

const apiBaseInput = document.getElementById("api-base");
apiBaseInput.value = getApiBase();

const modeSelect = document.getElementById("backend-mode");
const directModeRow = document.getElementById("direct-mode-row");
const sagemakerModeHint = document.getElementById("sagemaker-mode-hint");

function applyModeVisibility() {
  const isDirect = modeSelect.value === "direct";
  directModeRow.hidden = !isDirect;
  sagemakerModeHint.hidden = isDirect;
}

modeSelect.value = getMode();
applyModeVisibility();

modeSelect.addEventListener("change", () => {
  setMode(modeSelect.value);
  applyModeVisibility();
});

document.getElementById("save-api-base").addEventListener("click", () => {
  setApiBase(apiBaseInput.value || DEFAULT_API_BASE);
  apiBaseInput.value = getApiBase();
});

document.getElementById("check-health").addEventListener("click", async () => {
  const out = document.getElementById("health-output");
  out.hidden = false;
  out.textContent = "Checking...";
  const { ok, status, body } = await apiFetch("/health");
  showOutput(out, { status, body }, !ok);
});

// --- Tabs ---

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove("hidden");
  });
});

// --- /extract ---

document.getElementById("extract-submit").addEventListener("click", async () => {
  const out = document.getElementById("extract-output");
  out.hidden = false;

  const text = document.getElementById("extract-text").value;
  const maxTokens = parseInt(document.getElementById("extract-max-tokens").value, 10) || 512;
  const filingId = document.getElementById("extract-filing-id").value.trim();

  let result;
  if (getMode() === "sagemaker") {
    out.textContent = "Invoking SageMaker endpoint (may take a while on a cold start)...";
    result = await sagemakerInvoke("extract", { text, max_tokens: maxTokens });
  } else {
    out.textContent = "Submitting...";
    const payload = { text, max_tokens: maxTokens };
    if (filingId) payload.filing_id = filingId;
    result = await apiFetch("/extract", { method: "POST", body: JSON.stringify(payload) });
  }
  showOutput(out, { status: result.status, body: result.body }, !result.ok);
});

// --- /rag/query ---

document.getElementById("rag-submit").addEventListener("click", async () => {
  const out = document.getElementById("rag-output");
  out.hidden = false;

  const question = document.getElementById("rag-question").value;
  const topK = document.getElementById("rag-topk").value;

  let result;
  if (getMode() === "sagemaker") {
    out.textContent = "Invoking SageMaker endpoint (may take a while on a cold start)...";
    // Retrieval (embedding the question + querying intel.filing_sections)
    // isn't available in SageMaker mode -- the endpoint's inference.py has
    // no Postgres access. This sends the raw question as the prompt
    // directly, skipping RAG retrieval entirely; it demonstrates the model
    // invocation path, not the retrieval-augmented one.
    result = await sagemakerInvoke("rag_query", { prompt: question });
  } else {
    out.textContent = "Submitting...";
    const payload = { question };
    if (topK) payload.top_k = parseInt(topK, 10);
    result = await apiFetch("/rag/query", { method: "POST", body: JSON.stringify(payload) });
  }
  showOutput(out, { status: result.status, body: result.body }, !result.ok);
});

// --- /stats ---

document.getElementById("stats-refresh").addEventListener("click", async () => {
  const out = document.getElementById("stats-output");
  out.hidden = false;
  out.textContent = "Loading...";
  const { ok, status, body } = await apiFetch("/stats");
  showOutput(out, { status, body }, !ok);
});

// --- /pipeline/status ---

document.getElementById("pipeline-refresh").addEventListener("click", async () => {
  const out = document.getElementById("pipeline-output");
  out.hidden = false;
  out.textContent = "Loading...";
  const { ok, status, body } = await apiFetch("/pipeline/status");
  showOutput(out, { status, body }, !ok);
});
