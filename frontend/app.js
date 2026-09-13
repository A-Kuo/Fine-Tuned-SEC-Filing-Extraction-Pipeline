// FinDocAnalyzer Debug Console -- vanilla JS, no build step, no framework.
// Talks directly to the FastAPI backend (serving/api.py) via the API base
// URL below. The backend itself is not deployed here -- it needs a GPU/
// heavy ML deps this frontend's Vercel hosting can't and shouldn't provide.

const STORAGE_KEY = "findoc_debug_api_base";
const DEFAULT_API_BASE = "http://localhost:8000";

function getApiBase() {
  return localStorage.getItem(STORAGE_KEY) || DEFAULT_API_BASE;
}

function setApiBase(url) {
  localStorage.setItem(STORAGE_KEY, url.replace(/\/+$/, ""));
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

// --- Connection panel ---

const apiBaseInput = document.getElementById("api-base");
apiBaseInput.value = getApiBase();

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
  out.textContent = "Submitting...";

  const payload = {
    text: document.getElementById("extract-text").value,
    max_tokens: parseInt(document.getElementById("extract-max-tokens").value, 10) || 512,
  };
  const filingId = document.getElementById("extract-filing-id").value.trim();
  if (filingId) payload.filing_id = filingId;

  const { ok, status, body } = await apiFetch("/extract", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  showOutput(out, { status, body }, !ok);
});

// --- /rag/query ---

document.getElementById("rag-submit").addEventListener("click", async () => {
  const out = document.getElementById("rag-output");
  out.hidden = false;
  out.textContent = "Submitting...";

  const payload = { question: document.getElementById("rag-question").value };
  const topK = document.getElementById("rag-topk").value;
  if (topK) payload.top_k = parseInt(topK, 10);

  const { ok, status, body } = await apiFetch("/rag/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  showOutput(out, { status, body }, !ok);
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
