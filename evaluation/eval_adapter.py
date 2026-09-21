"""Score a trained adapter against the base model on held-out examples, going
through the same path the API serves: ExtractionEngine (prompt building,
FinancialLLM batch generation, the 5-stage parser, schema validation).

What this measures, and what it does not
----------------------------------------
The held-out examples come from the SAME synthetic generator and templates as
the training set (scripts/download_dataset.py). A high score therefore shows
the adapter learned the task on in-distribution data; it says nothing about
accuracy on real SEC filings, which this script does not evaluate
(`real_filing_eval` is recorded as "not run"). Results are reported as counts
(k of n), never rounded percentages: with a few dozen examples a percentage
would look far more precise than it is.

The base model and the adapter are each loaded through the project's own
FinancialLLM.from_pretrained in turn, not toggled on one model with PEFT,
because serving merges the adapter into 4-bit weights and merged weights
can differ from an unmerged adapter. Measuring the merged path measures what
is actually served.

Usage (GPU required):
    python evaluation/eval_adapter.py --adapter models/llama-sec-v1 \\
        --out evaluation/results/adapter_eval.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from evaluation.metrics import (
    ALL_FIELDS,
    error_taxonomy_breakdown,
    exact_json_match_rate,
    null_handling_correctness,
    per_field_accuracy,
)
from src.extraction.inference import ExtractionEngine, ExtractionRequest

DATA_PROVENANCE = (
    "Synthetic held-out examples from the same generator and templates as the "
    "training set (scripts/download_dataset.py). Measures how well the adapter "
    "learned the task on in-distribution data; NOT accuracy on real SEC filings."
)
FRESH_EXAMPLES_SEED = 1234
FAILURE_SAMPLE_CHARS = 400
MAX_FAILURE_SAMPLES = 3


# ─── Loading examples ────────────────────────────────────────────────────────

def load_examples(path: str | Path) -> list[dict]:
    """Read a JSONL file of {input, output, ...} rows into {input, truth}."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            truth = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]
            examples.append({"id": row.get("id"), "input": row["input"], "truth": truth})
    return examples


def generate_fresh_examples(count: int, seed: int = FRESH_EXAMPLES_SEED) -> list[dict]:
    """Extra held-out examples from the training generator with a different
    seed than the train (42) and test (43) splits."""
    if count <= 0:
        return []
    from scripts.download_dataset import generate_dataset

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fresh.jsonl"
        generate_dataset(count, path, seed=seed)
        return load_examples(path)


def drop_training_overlap(examples: list[dict], train_path: str | Path | None) -> tuple[list[dict], int]:
    """Remove any held-out example whose input text also appears in the
    training set, so nothing being scored was trained on."""
    if not train_path or not Path(train_path).exists():
        return examples, 0
    seen = {ex["input"] for ex in load_examples(train_path)}
    kept = [ex for ex in examples if ex["input"] not in seen]
    return kept, len(examples) - len(kept)


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _fully_correct(pred: dict, truth: dict) -> bool:
    return exact_json_match_rate([pred], [truth]) == 1.0


def evaluate_engine(engine: ExtractionEngine, examples: list[dict], batch_size: int) -> dict:
    """Run every example through engine.extract_batch (the API's path) and
    score against ground truth. Counts only; see the module docstring."""
    requests = [ExtractionRequest(text=ex["input"]) for ex in examples]
    responses = []
    for start in range(0, len(requests), batch_size):
        responses.extend(engine.extract_batch(requests[start:start + batch_size]))

    predictions: list[dict] = []
    stages: list[str | None] = []
    status_counts: dict[str, int] = {}
    latencies: list[float] = []
    failures: list[dict] = []

    for example, response in zip(examples, responses):
        predictions.append(response.result.to_dict() if response.result else {})
        stages.append(response.telemetry.winning_stage if response.telemetry else None)
        status_counts[response.status] = status_counts.get(response.status, 0) + 1
        latencies.append(response.latency_ms)
        if response.status != "success" and len(failures) < MAX_FAILURE_SAMPLES:
            failures.append({
                "id": example.get("id"),
                "status": response.status,
                "error": response.error,
                "raw_output_head": (response.raw_output or "")[:FAILURE_SAMPLE_CHARS],
            })

    truths = [ex["truth"] for ex in examples]
    n = len(examples)
    per_field = {
        field: {"correct": counts["correct"], "total": counts["total"]}
        for field, counts in per_field_accuracy(predictions, truths).items()
    }

    stage_counts: dict[str, int] = {}
    for stage in stages:
        key = stage or "none"
        stage_counts[key] = stage_counts.get(key, 0) + 1

    return {
        "n": n,
        "status_counts": status_counts,
        "json_parsed": sum(1 for r in responses if r.result is not None),
        "schema_valid": status_counts.get("success", 0),
        "fully_correct": sum(1 for p, t in zip(predictions, truths) if _fully_correct(p, t)),
        "per_field": per_field,
        "null_handling": null_handling_correctness(predictions, truths),
        "error_taxonomy": error_taxonomy_breakdown(predictions, truths, stages),
        "parser_winning_stage": stage_counts,
        "mean_latency_ms": round(sum(latencies) / n, 1) if n else None,
        "failure_samples": failures,
    }


def compare(base: dict, adapter: dict) -> dict:
    """Side-by-side counts, so the report answers 'what did fine-tuning add?'"""
    return {
        "n": adapter["n"],
        "json_parsed": {"base": base["json_parsed"], "adapter": adapter["json_parsed"]},
        "schema_valid": {"base": base["schema_valid"], "adapter": adapter["schema_valid"]},
        "fully_correct": {"base": base["fully_correct"], "adapter": adapter["fully_correct"]},
        "per_field_correct": {
            field: {
                "base": base["per_field"][field]["correct"],
                "adapter": adapter["per_field"][field]["correct"],
                "of": adapter["per_field"][field]["total"],
            }
            for field in ALL_FIELDS
        },
    }


# ─── Model loading (one model in GPU memory at a time) ───────────────────────

@contextmanager
def _local_adapter_only():
    """FinancialLLM.from_pretrained prefers a registry-hosted adapter when a
    DagsHub token is set; the point here is to score the LOCAL adapter under
    test, so the registry lookup is switched off while loading."""
    saved = os.environ.pop("DAGSHUB_USER_TOKEN", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["DAGSHUB_USER_TOKEN"] = saved


def default_loader(adapter_path: str | None):
    """Load through the serving path. adapter_path=None loads the base model
    alone (from_pretrained warns and uses base weights when no adapter exists)."""
    from src.extraction.model import FinancialLLM

    target = adapter_path or str(Path(tempfile.gettempdir()) / "no-adapter-here")
    with _local_adapter_only():
        return FinancialLLM.from_pretrained(adapter_path=target)


def _release(llm) -> None:
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_evaluation(
    adapter_dir: str,
    examples: list[dict],
    batch_size: int = 4,
    load_llm: Callable[[str | None], object] = default_loader,
    overlap_removed: int = 0,
) -> dict:
    """Score base then adapter on the same examples and return the report."""
    llm = load_llm(None)
    base_version = getattr(llm, "model_version", "unknown")
    logger.info(f"Scoring base model on {len(examples)} examples...")
    base = evaluate_engine(ExtractionEngine(model=llm), examples, batch_size)
    _release(llm)

    llm = load_llm(adapter_dir)
    adapter_version = getattr(llm, "model_version", "unknown")
    logger.info(f"Scoring adapter ({adapter_dir}) on {len(examples)} examples...")
    adapter = evaluate_engine(ExtractionEngine(model=llm), examples, batch_size)
    _release(llm)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": adapter_dir,
        "data_provenance": DATA_PROVENANCE,
        "real_filing_eval": "not run",
        "reporting": "counts (k of n), not percentages; small n",
        "n_examples": len(examples),
        "overlap_with_training_removed": overlap_removed,
        "base_model_version": base_version,
        "adapter_model_version": adapter_version,
        "base": base,
        "adapter": adapter,
        "comparison": compare(base, adapter),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Base-vs-adapter evaluation on held-out synthetic examples")
    p.add_argument("--adapter", default="models/llama-sec-v1", help="Trained adapter directory")
    p.add_argument("--test-path", default=None, help="Held-out JSONL (default: config data.test_path)")
    p.add_argument("--train-path", default=None, help="Training JSONL, used to drop any overlap (default: config data.train_path)")
    p.add_argument("--extra", type=int, default=20, help="Extra fresh held-out examples to generate")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N examples (smoke runs)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--out", default="evaluation/results/adapter_eval.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.core.config import get_project_root, load_config

    config = load_config()
    root = get_project_root()
    test_path = args.test_path or root / config["data"]["test_path"]
    train_path = args.train_path or root / config["data"]["train_path"]

    examples = load_examples(test_path) + generate_fresh_examples(args.extra)
    examples, removed = drop_training_overlap(examples, train_path)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise SystemExit("No held-out examples to evaluate.")

    report = run_evaluation(args.adapter, examples, args.batch_size, overlap_removed=removed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cmp = report["comparison"]
    n = cmp["n"]
    print(f"Held-out synthetic examples: {n} (real-filing eval: not run)")
    for key in ("json_parsed", "schema_valid", "fully_correct"):
        print(f"  {key:<14} base {cmp[key]['base']}/{n}   adapter {cmp[key]['adapter']}/{n}")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
