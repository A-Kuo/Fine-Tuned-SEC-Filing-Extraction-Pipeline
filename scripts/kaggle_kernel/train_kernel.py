"""Entry point that runs INSIDE a Kaggle Notebook (kernel).

Pushed and triggered by scripts/submit_kaggle_job.py. Clones the repo,
installs dependencies, generates training data, trains, then (in a full run)
scores the adapter against the base model and publishes it. Every phase after
training is recorded in run_report.json but cannot fail an otherwise
successful training run; a failed training step does fail the kernel.

Run modes (RUN_MODE below is rewritten at push time by
scripts/submit_kaggle_job.py --mode):
    smoke  8 examples, 1 epoch, a 2-example evaluation, no upload. Minutes,
           and it exercises every stage: gated-model access, secrets, the TRL
           pin, GPU memory, tracking, evaluation.
    full   the whole training set, the base-vs-adapter evaluation, and the
           Hugging Face upload.

Secrets (Kaggle Notebook -> Add-ons -> Secrets), all optional except HF_TOKEN:
    HF_TOKEN             HuggingFace token for the gated base-model download
    HF_WRITE_TOKEN       write-scoped token for publishing the adapter
    HF_ADAPTER_REPO      owner/name of the (private) adapter repo
    MLFLOW_TRACKING_URI  remote MLFlow server (default: local file store)
    DAGSHUB_USER_TOKEN   only if using DagsHub tracking

Data generation: config.yaml's kaggle.dataset_id is empty, so
training/train.py's resolve_dataset_path() falls back to
kaggle.local_fallback ("data/sec_filings_train.jsonl") -- a file that is
gitignored and therefore absent from this script's fresh `git clone`.
scripts/download_dataset.py + scripts/format_data.py (pure local generation)
produce it, mirroring notebooks/train_qlora.ipynb's cell 9.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_MODE = "full"  # rewritten by scripts/submit_kaggle_job.py ("smoke" or "full")

REPO_URL = "https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
REPO_DIR = "/kaggle/working/repo"
REPORT_DIR = os.environ.get("KERNEL_REPORT_DIR", "/kaggle/working/reports")
ADAPTER_DIR = "models/llama-sec-v1"

# The version the notebooks were validated against; requirements.txt only sets
# lower bounds, so Kaggle would otherwise install whatever is newest.
TRL_PIN = "trl==0.12.2"

SECRET_NAMES = (
    "HF_TOKEN",
    "HF_WRITE_TOKEN",
    "HF_ADAPTER_REPO",
    "MLFLOW_TRACKING_URI",
    "DAGSHUB_USER_TOKEN",
)

MODES = {
    "smoke": {
        "train_args": ["--max_samples", "8", "--num_epochs", "1"],
        "eval_args": ["--extra", "0", "--limit", "2"],
        "upload": False,
    },
    "full": {
        "train_args": [],
        "eval_args": ["--extra", "20"],
        "upload": True,
    },
}


def _load_kaggle_secrets() -> dict[str, str]:
    """Populate os.environ from Kaggle's UserSecretsClient and report, by
    name only, what happened to each secret (values are never printed).

    This used to swallow every failure silently, so a secret that wasn't
    attached to the notebook looked identical to one that loaded fine, and the
    first symptom was a 401 from Hugging Face minutes later.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        status = {
            name: "present in environment" if os.environ.get(name) else "not set"
            for name in SECRET_NAMES
        }
        print("[kernel] secrets: not running on Kaggle; using the existing environment", flush=True)
    else:
        status = {}
        try:
            client = UserSecretsClient()
        except Exception as e:
            client = None
            status = {name: f"not loaded ({type(e).__name__}: {str(e)[:120]})" for name in SECRET_NAMES}
        if client is not None:
            for key in SECRET_NAMES:
                try:
                    os.environ[key] = client.get_secret(key)
                    status[key] = "loaded"
                except Exception as e:
                    status[key] = f"not loaded ({type(e).__name__}: {str(e)[:120]})"

    for key, outcome in status.items():
        print(f"[kernel] secret {key}: {outcome}", flush=True)
    return status


def _http_status(error: Exception) -> int | None:
    return getattr(getattr(error, "response", None), "status_code", None)


def _preflight_hf_access(model_id: str = BASE_MODEL) -> tuple[bool, str]:
    """Check, in seconds, that the gated base model is reachable -- before the
    ~5 minutes of pip installs, not after them.

    Returns (ok, message). Only a definite auth/gating failure returns
    ok=False; a network blip or a missing library returns ok=True with a
    note, so this can never block a run that would have worked. Tells apart
    the three real causes: no token at all, a token Hugging Face rejects, and
    a valid token whose account lacks access to the model.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, (
            "HF_TOKEN is not set. Add a Hugging Face read token as a Kaggle secret named exactly "
            "HF_TOKEN and tick it in THIS notebook's Add-ons > Secrets panel (a secret has to be "
            "attached to each notebook; saving it on the account is not enough)."
        )

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        return True, "huggingface_hub is not importable; skipping the access check"

    try:
        username = HfApi(token=token).whoami().get("name", "unknown")
    except Exception as e:
        if _http_status(e) in (401, 403):
            return False, (
                "HF_TOKEN was rejected by Hugging Face (invalid, expired or revoked). "
                "Create a new read token at https://huggingface.co/settings/tokens and update the Kaggle secret."
            )
        return True, f"could not verify HF_TOKEN ({type(e).__name__}); continuing"

    try:
        hf_hub_download(repo_id=model_id, filename="config.json", token=token)
    except Exception as e:
        if _http_status(e) in (401, 403) or type(e).__name__ == "GatedRepoError":
            return False, (
                f"HF_TOKEN is valid (Hugging Face account '{username}') but that account has no access to "
                f"{model_id}. Accept the license at https://huggingface.co/{model_id} while signed in as "
                f"'{username}'. If the token is fine-grained, also enable 'Read access to contents of all "
                "public gated repos you can access'."
            )
        return True, f"could not check access to {model_id} ({type(e).__name__}); continuing"

    return True, f"Hugging Face access OK (account '{username}', {model_id})"


def _run(cmd: list[str]) -> None:
    """A step whose failure must stop the run (prerequisites and training)."""
    print(f"[kernel] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_DIR, check=True, env=os.environ)


def _run_step(name: str, cmd: list[str], report: dict) -> bool:
    """A post-training step: failure is recorded and printed, never raised."""
    print(f"[kernel] step {name}: {' '.join(cmd)}", flush=True)
    started = time.time()
    try:
        subprocess.run(cmd, cwd=REPO_DIR, check=True, env=os.environ)
    except (subprocess.CalledProcessError, OSError) as e:
        report["steps"][name] = {"status": "failed", "error": str(e), "seconds": round(time.time() - started, 1)}
        print(f"[kernel] step {name} FAILED (training result is unaffected): {e}", flush=True)
        return False
    report["steps"][name] = {"status": "ok", "seconds": round(time.time() - started, 1)}
    return True


def _write_report(report: dict) -> None:
    try:
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (Path(REPORT_DIR) / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[kernel] could not write run_report.json: {e}", flush=True)


def _remove_checkpoints() -> str:
    """Trainer checkpoints (with optimizer state) are useless once the final
    adapter is saved and would bloat the kernel output that GitHub Actions
    downloads. Best-effort."""
    removed = 0
    for path in (Path(REPO_DIR) / ADAPTER_DIR).glob("checkpoint-*"):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return f"removed {removed} checkpoint dir(s)"


def main(mode: str | None = None) -> None:
    mode = mode or RUN_MODE
    if mode not in MODES:
        raise SystemExit(f"Unknown RUN_MODE {mode!r}; expected one of {sorted(MODES)}")
    plan = MODES[mode]

    report = {
        "schema_version": 1,
        "mode": mode,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    report["secrets"] = _load_kaggle_secrets()

    # Fail now, with a specific reason, rather than ~5 minutes into the pip
    # installs. Both modes need the gated base model, so this is fatal in both.
    ok, message = _preflight_hf_access()
    report["steps"]["preflight"] = {"status": "ok" if ok else "failed", "detail": message}
    print(f"[kernel] preflight: {message}", flush=True)
    if not ok:
        _write_report(report)
        raise SystemExit(f"Preflight failed: {message}")

    stage = "setup"
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
        _run([sys.executable, "-m", "pip", "install", "-q", TRL_PIN])
        _run([sys.executable, "scripts/download_dataset.py"])
        _run([sys.executable, "scripts/format_data.py"])

        _run_step(
            "environment",
            [sys.executable, "scripts/kaggle_kernel/collect_environment.py", "--out", f"{REPORT_DIR}/environment.json"],
            report,
        )

        stage = "training"
        _run([sys.executable, "training/train.py", *plan["train_args"]])
        report["steps"]["training"] = {"status": "ok"}
        report["steps"]["cleanup"] = {"status": "ok", "detail": _remove_checkpoints()}
    except (subprocess.CalledProcessError, OSError) as e:
        report["steps"][stage] = {"status": "failed", "error": str(e)}
        _write_report(report)
        raise

    _run_step(
        "evaluation",
        [
            sys.executable, "evaluation/eval_adapter.py",
            "--adapter", ADAPTER_DIR,
            "--out", f"{REPORT_DIR}/adapter_eval.json",
            *plan["eval_args"],
        ],
        report,
    )

    if plan["upload"]:
        _run_step(
            "hf_upload",
            [sys.executable, "scripts/upload_adapter_hf.py", "--adapter", ADAPTER_DIR, "--out", f"{REPORT_DIR}/hf_upload.json"],
            report,
        )
    else:
        report["steps"]["hf_upload"] = {"status": "skipped", "reason": f"{mode} mode never uploads"}

    _write_report(report)


if __name__ == "__main__":
    main()
