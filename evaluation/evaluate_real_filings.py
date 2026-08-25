from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def evaluate_section_coverage(predictions, labels):
    total = len(labels)
    hits = 0
    for pred, gold in zip(predictions, labels):
        pred_sections = {s["section_type"] for s in pred.get("sections", [])}
        gold_sections = set(gold.get("expected_sections", []))
        if gold_sections.issubset(pred_sections):
            hits += 1
    return hits / total if total else 0.0


def main():
    pred_path = Path("evaluation/predictions.jsonl")
    gold_path = Path("evaluation/gold/sample_labels.jsonl")

    predictions = list(load_jsonl(pred_path))
    labels = list(load_jsonl(gold_path))

    coverage = evaluate_section_coverage(predictions, labels)

    results = {
        "section_coverage": round(coverage, 4),
        "num_predictions": len(predictions),
        "num_labels": len(labels),
    }

    out_path = Path("evaluation/results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()