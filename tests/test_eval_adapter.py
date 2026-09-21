"""Tests for evaluation/eval_adapter.py. A fake model stands in for the GPU
model: each example carries a unique marker (EX-000...) that the fake finds in
the prompt to decide what to answer, so scoring, comparison and report shape
are all exercised with no torch, no downloads and no GPU."""

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.eval_adapter import (
    DATA_PROVENANCE,
    _local_adapter_only,
    compare,
    drop_training_overlap,
    evaluate_engine,
    generate_fresh_examples,
    load_examples,
    run_evaluation,
)
from evaluation.metrics import ALL_FIELDS
from src.extraction.inference import ExtractionEngine


def _truth(i: int) -> dict:
    return {
        "filing_id": f"000{i}-24-00000{i}",
        "company_name": f"Example Corp {i}",
        "ticker": f"EX{i}",
        "filing_type": "10-K",
        "date": f"2024-0{i + 1}-15",
        "fiscal_year_end": f"2023-12-3{i % 2}",
        "revenue": f"${10 + i}.5 billion",
        "net_income": f"${i + 1}.2 billion",
        "total_assets": f"${50 + i}.0 billion",
        "total_liabilities": f"${20 + i}.0 billion",
        "eps": f"${i + 1}.10",
        "sector": "Technology",
    }


def _examples(n: int = 4) -> list[dict]:
    return [
        {"id": f"ex-{i}", "input": f"Filing document marker EX-{i:03d} annual report text.", "truth": _truth(i)}
        for i in range(n)
    ]


class FakeTokenizer:
    chat_template = None


class FakeLLM:
    """Answers each prompt by the EX-nnn marker it contains."""

    def __init__(self, respond, version="fake-v1"):
        self.tokenizer = FakeTokenizer()
        self.model_version = version
        self._respond = respond
        self.prompts_seen = 0

    def generate_batch(self, prompts, max_tokens=None):
        self.prompts_seen += len(prompts)
        return [(self._respond(p), 12.0) for p in prompts]


def _marker(prompt: str) -> int:
    return int(re.search(r"EX-(\d{3})", prompt).group(1))


def perfect(prompt: str) -> str:
    return json.dumps(_truth(_marker(prompt)))


def garbage(prompt: str) -> str:
    return "complete garbage, no structure whatsoever"


def wrong_revenue(prompt: str) -> str:
    truth = _truth(_marker(prompt))
    truth["revenue"] = "$999.9 billion"
    return json.dumps(truth)


class TestEvaluateEngine:
    def test_perfect_model_gets_everything_right(self):
        report = evaluate_engine(ExtractionEngine(model=FakeLLM(perfect)), _examples(4), batch_size=2)

        assert report["n"] == 4
        assert report["json_parsed"] == 4
        assert report["schema_valid"] == 4
        assert report["fully_correct"] == 4
        for field, counts in report["per_field"].items():
            assert counts["correct"] == counts["total"] == 4, field
        assert report["status_counts"] == {"success": 4}

    def test_unparseable_output_scores_zero_and_keeps_failure_samples(self):
        report = evaluate_engine(ExtractionEngine(model=FakeLLM(garbage)), _examples(4), batch_size=4)

        assert report["json_parsed"] == 0
        assert report["schema_valid"] == 0
        assert report["fully_correct"] == 0
        assert report["parser_winning_stage"] == {"none": 4}
        assert 1 <= len(report["failure_samples"]) <= 3
        assert "complete garbage" in report["failure_samples"][0]["raw_output_head"]

    def test_one_wrong_field_costs_exactly_one_fully_correct_and_one_field(self):
        report = evaluate_engine(ExtractionEngine(model=FakeLLM(wrong_revenue)), _examples(4), batch_size=4)

        assert report["json_parsed"] == 4
        assert report["fully_correct"] == 0
        assert report["per_field"]["revenue"]["correct"] == 0
        assert report["per_field"]["revenue"]["total"] == 4
        assert report["per_field"]["company_name"]["correct"] == 4

    def test_batches_cover_every_example_exactly_once(self):
        llm = FakeLLM(perfect)
        evaluate_engine(ExtractionEngine(model=llm), _examples(7), batch_size=3)
        assert llm.prompts_seen == 7

    def test_reports_counts_not_percentages(self):
        report = evaluate_engine(ExtractionEngine(model=FakeLLM(perfect)), _examples(2), batch_size=2)
        assert all(set(c) == {"correct", "total"} for c in report["per_field"].values())
        assert isinstance(report["fully_correct"], int)


class TestCompare:
    def test_side_by_side_counts(self):
        base = evaluate_engine(ExtractionEngine(model=FakeLLM(garbage)), _examples(3), batch_size=3)
        adapter = evaluate_engine(ExtractionEngine(model=FakeLLM(perfect)), _examples(3), batch_size=3)

        result = compare(base, adapter)

        assert result["n"] == 3
        assert result["fully_correct"] == {"base": 0, "adapter": 3}
        assert result["schema_valid"] == {"base": 0, "adapter": 3}
        assert set(result["per_field_correct"]) == set(ALL_FIELDS)
        assert result["per_field_correct"]["revenue"] == {"base": 0, "adapter": 3, "of": 3}


class TestRunEvaluation:
    def test_scores_base_then_adapter_and_labels_what_it_measures(self):
        loads = []

        def loader(adapter_path):
            loads.append(adapter_path)
            return FakeLLM(garbage, "base-v") if adapter_path is None else FakeLLM(perfect, "adapter-v")

        report = run_evaluation("models/llama-sec-v1", _examples(4), batch_size=2, load_llm=loader, overlap_removed=1)

        assert loads == [None, "models/llama-sec-v1"]
        assert report["base_model_version"] == "base-v"
        assert report["adapter_model_version"] == "adapter-v"
        assert report["comparison"]["fully_correct"] == {"base": 0, "adapter": 4}
        assert report["n_examples"] == 4
        assert report["overlap_with_training_removed"] == 1
        assert report["real_filing_eval"] == "not run"
        assert report["data_provenance"] == DATA_PROVENANCE
        assert "NOT accuracy on real SEC filings" in report["data_provenance"]

    def test_report_is_json_serializable(self):
        def loader(adapter_path):
            return FakeLLM(perfect)

        report = run_evaluation("a", _examples(2), load_llm=loader)
        assert json.loads(json.dumps(report))["n_examples"] == 2


class TestLoadingAndOverlap:
    def test_load_examples_parses_the_output_column(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            json.dumps({"id": "a", "input": "text A", "output": json.dumps({"ticker": "AAA"})}) + "\n\n"
            + json.dumps({"id": "b", "input": "text B", "output": {"ticker": "BBB"}}) + "\n"
        )
        loaded = load_examples(path)
        assert [e["truth"]["ticker"] for e in loaded] == ["AAA", "BBB"]
        assert [e["input"] for e in loaded] == ["text A", "text B"]

    def test_examples_that_appear_in_training_are_dropped(self, tmp_path):
        train = tmp_path / "train.jsonl"
        train.write_text(json.dumps({"id": "t", "input": "seen in training", "output": "{}"}) + "\n")
        examples = [
            {"id": "1", "input": "seen in training", "truth": {}},
            {"id": "2", "input": "brand new", "truth": {}},
        ]

        kept, removed = drop_training_overlap(examples, train)

        assert [e["id"] for e in kept] == ["2"]
        assert removed == 1

    def test_no_training_file_means_nothing_is_dropped(self, tmp_path):
        examples = [{"id": "1", "input": "x", "truth": {}}]
        assert drop_training_overlap(examples, tmp_path / "missing.jsonl") == (examples, 0)
        assert drop_training_overlap(examples, None) == (examples, 0)

    def test_fresh_examples_are_deterministic_and_well_formed(self):
        first = generate_fresh_examples(3, seed=1234)
        second = generate_fresh_examples(3, seed=1234)
        other = generate_fresh_examples(3, seed=42)

        assert len(first) == 3
        assert [e["input"] for e in first] == [e["input"] for e in second]
        assert [e["input"] for e in first] != [e["input"] for e in other]
        assert set(first[0]["truth"]) == set(ALL_FIELDS)

    def test_zero_fresh_examples_generates_nothing(self):
        assert generate_fresh_examples(0) == []


class TestRegistryIsSwitchedOffWhileLoading:
    def test_dagshub_token_is_hidden_during_the_load_and_restored_after(self, monkeypatch):
        monkeypatch.setenv("DAGSHUB_USER_TOKEN", "secret")

        with _local_adapter_only():
            assert "DAGSHUB_USER_TOKEN" not in os.environ

        assert os.environ["DAGSHUB_USER_TOKEN"] == "secret"

    def test_no_token_stays_unset(self, monkeypatch):
        monkeypatch.delenv("DAGSHUB_USER_TOKEN", raising=False)
        with _local_adapter_only():
            pass
        assert "DAGSHUB_USER_TOKEN" not in os.environ


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
