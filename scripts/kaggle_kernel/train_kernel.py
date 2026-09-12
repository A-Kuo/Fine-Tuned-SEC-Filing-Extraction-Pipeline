"""Entry point that runs INSIDE a Kaggle Notebook (kernel).

Pushed and triggered by scripts/submit_kaggle_job.py. Clones the repo,
installs dependencies, generates training data, and runs the same
training/train.py used for local training, so both paths log to the same
MLFlow (DagsHub) experiment.

Secrets required (set via Kaggle Notebook -> Add-ons -> Secrets):
    DAGSHUB_USER_TOKEN   - authenticates MLFlow tracking to DagsHub
    HF_TOKEN             - HuggingFace token for base model download

Data generation: config.yaml's kaggle.dataset_id is empty, so
training/train.py's resolve_dataset_path() falls back to
kaggle.local_fallback ("data/sec_filings_train.jsonl") -- a file that is
gitignored and therefore absent from this script's fresh `git clone`.
Without generating it here first, training used to crash immediately with
FileNotFoundError. scripts/download_dataset.py + scripts/format_data.py
(pure local generation, no network/credentials beyond what's already
available) produce it, mirroring notebooks/train_qlora.ipynb's cell 9,
which already does this successfully for the separate Colab path.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/A-Kuo/Fine-Tuned-SEC-Filing-Extraction-Pipeline"
REPO_DIR = "/kaggle/working/repo"


def _load_kaggle_secrets() -> None:
    """Populate os.environ from Kaggle's UserSecretsClient, if available."""
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        for key in ("DAGSHUB_USER_TOKEN", "HF_TOKEN"):
            try:
                os.environ[key] = secrets.get_secret(key)
            except Exception:
                pass
    except ImportError:
        pass


def main() -> None:
    _load_kaggle_secrets()

    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=REPO_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/download_dataset.py"],
        cwd=REPO_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/format_data.py"],
        cwd=REPO_DIR,
        check=True,
    )
    subprocess.run(
        [sys.executable, "training/train.py"],
        cwd=REPO_DIR,
        check=True,
        env=os.environ,
    )


if __name__ == "__main__":
    main()
