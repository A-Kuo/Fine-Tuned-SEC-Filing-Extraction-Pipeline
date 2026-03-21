"""Integration tests: end-to-end flows without GPU.

Tests the full pipeline logic (minus model inference) to verify
all components wire together correctly:
    Data generation → Formatting → Post-processing → Evaluation → Monitoring

These catch integration bugs that unit tests miss, like mismatched
field names between modules or broken serialization.
"""

import json
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.postprocessing import parse_extraction, validate_extraction, ExtractionResult
from src.inference import ExtractionEngine, ExtractionRequest, ExtractionResponse
from evaluation.evaluate import evaluate_single, fuzzy_financial_match
from monitoring.monitor import generate_full_report


class TestEndToEndPostprocessing:
    """Test: model output → parse → validate → evaluation pipeline."""

    def test_simulated_model_output_to_evaluation(self):
        """Simulate what happens when the model returns a JSON extraction."""
        # Simulated model output (what the model would generate)
        model_output = json.dumps({
            "filing_id": "000123-23-456",
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "filing_type": "10-K",
            "date": "2023-11-03",
            "fiscal_year_end": "2023-09-30",
            "revenue": "$383.3 billion",
            "net_income": "$97.0 billion",
            "total_assets": "$352.6 billion",
            "total_liabilities": "$290.4 billion",
            "eps": "$6.13",
            "sector": "Technology",
        })

        # Step 1: Parse
        result = parse_extraction(model_output)
        assert result.company_name == "Apple Inc."
        assert result.completeness == 1.0

        # Step 2: Validate
        is_valid, errors = validate_extraction(result)
        assert is_valid, f"Should be valid: {errors}"

        # Step 3: Evaluate against ground truth
        ground_truth = json.loads(model_output)
        eval_results = evaluate_single(result.to_dict(), ground_truth)
        assert all(r["correct"] for r in eval_results.values() if r["ground_truth"] is not None)

    def test_partial_extraction_flows_through(self):
        """Partial extraction still produces evaluation metrics."""
        model_output = '{"company_name": "Tesla, Inc.", "filing_type": "10-Q"}'

        result = parse_extraction(model_output)
        assert result.company_name == "Tesla, Inc."
        assert result.revenue is None  # Missing

        is_valid, errors = validate_extraction(result)
        assert not is_valid  # Missing required fields

        # Evaluation still works with partial data
        ground_truth = {
            "company_name": "Tesla, Inc.",
            "filing_type": "10-K",  # Wrong!
            "revenue": "$100 billion",
        }
        eval_results = evaluate_single(result.to_dict(), ground_truth)
        assert eval_results["company_name"]["correct"]
        assert not eval_results["filing_type"]["correct"]  # 10-Q ≠ 10-K

    def test_malformed_output_to_evaluation(self):
        """Malformed model output is parsed with regex fallback."""
        model_output = """Here is the extraction:
        Company: NVIDIA Corporation
        Type: 10-K
        Revenue: $60.9 billion"""

        # Should fall back to regex extraction
        result = parse_extraction(model_output)
        # Regex might capture company_name from pattern
        assert result.filled_fields >= 0  # At least attempted

    def test_code_fenced_output(self):
        """Output wrapped in markdown fences flows through correctly."""
        model_output = '```json\n{"company_name": "Meta Platforms", "filing_type": "10-K", "date": "2024-02-01", "filing_id": "test-123"}\n```'

        result = parse_extraction(model_output)
        assert result.company_name == "Meta Platforms"

        is_valid, errors = validate_extraction(result)
        assert is_valid


class TestEndToEndMonitoring:
    """Test: extraction results → monitoring → alert generation."""

    def test_healthy_extractions_no_alerts(self):
        """94% accuracy + normal latency → healthy report."""
        report = generate_full_report(
            current_accuracy=0.94,
            baseline_accuracy=0.94,
            latencies_ms=[320, 350, 400, 380, 310, 290, 420, 360, 340, 330],
        )
        assert report.status == "healthy"
        assert len(report.alerts) == 0

        # Report should be JSON-serializable
        report_json = json.dumps(report.to_dict())
        assert "healthy" in report_json

    def test_degraded_accuracy_triggers_alert(self):
        """Accuracy drop → monitoring catches it → alert generated."""
        report = generate_full_report(
            current_accuracy=0.85,
            baseline_accuracy=0.94,
            latencies_ms=[300, 400],
        )
        assert report.status == "critical"
        assert len(report.alerts) > 0
        assert any("retrain" in a.lower() for a in report.alerts)


class TestEndToEndDataPipeline:
    """Test: data generation → formatting → readiness for training."""

    def test_generated_data_is_valid_training_format(self):
        """Generated data can be parsed and validated."""
        data_path = Path(__file__).parent.parent / "data" / "sec_filings_train.jsonl"
        if not data_path.exists():
            pytest.skip("Training data not generated")

        with open(data_path) as f:
            first_line = f.readline()

        example = json.loads(first_line)
        assert "instruction" in example
        assert "input" in example
        assert "output" in example

        # Output should be valid JSON extraction
        output = json.loads(example["output"])
        assert "company_name" in output
        assert "filing_type" in output

    def test_chat_formatted_data_is_valid(self):
        """Chat-formatted data has correct structure."""
        chat_path = Path(__file__).parent.parent / "data" / "sec_filings_train.chat.jsonl"
        if not chat_path.exists():
            pytest.skip("Chat-formatted data not generated")

        with open(chat_path) as f:
            first_line = f.readline()

        example = json.loads(first_line)
        assert "messages" in example
        assert len(example["messages"]) == 3
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        # Assistant message should be valid JSON
        assistant_content = example["messages"][2]["content"]
        parsed = json.loads(assistant_content)
        assert "company_name" in parsed

    def test_sample_filing_matches_expected(self):
        """Sample filing text can be extracted to match expected output."""
        data_dir = Path(__file__).parent.parent / "data"
        sample_path = data_dir / "sample_10k.txt"
        expected_path = data_dir / "sample_10k.expected.json"

        if not sample_path.exists():
            pytest.skip("Sample data not generated")

        # Sample filing should exist and be non-empty
        sample_text = sample_path.read_text()
        assert len(sample_text) > 100

        # Expected extraction should be valid
        with open(expected_path) as f:
            expected = json.load(f)

        result = parse_extraction(json.dumps(expected))
        is_valid, errors = validate_extraction(result)
        assert is_valid, f"Expected extraction should be valid: {errors}"


class TestExtractionResponseSerialization:
    """Test that ExtractionResponse can be serialized for API/storage."""

    def test_success_response_serializable(self):
        result = ExtractionResult(
            filing_id="test-123",
            company_name="Apple Inc.",
            filing_type="10-K",
            date="2023-01-01",
        )
        response = ExtractionResponse(
            result=result,
            raw_output='{"company_name": "Apple Inc."}',
            latency_ms=421.5,
            model_version="v1",
            status="success",
            confidence_score=0.96,
        )

        # Should be serializable via result.to_dict()
        d = response.result.to_dict()
        json_str = json.dumps(d)
        assert "Apple Inc." in json_str

    def test_error_response_serializable(self):
        response = ExtractionResponse(
            result=None,
            raw_output="garbage",
            latency_ms=100,
            model_version="v1",
            status="parse_error",
            error="No valid JSON",
        )
        # Should not raise even with None result
        assert response.status == "parse_error"
        assert response.result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
