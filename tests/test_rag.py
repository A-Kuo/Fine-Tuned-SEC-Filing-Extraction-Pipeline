"""Tests for the lightweight RAG demo (src/rag/).

Every piece here is fully testable with fakes -- no real Postgres
connection and no real embedding/generation model is loaded. retrieve()
takes a NormalizedStorage with a mocked _connection (exact pattern from
tests/test_normalized_storage.py); generate_answer() takes an injected
generate_fn instead of loading FinancialLLM itself.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.embed import Embedder
from src.rag.generate import generate_answer
from src.rag.retrieve import retrieve
from src.storage.normalized_storage import NormalizedStorage


def _make_storage() -> NormalizedStorage:
    storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
    storage._available = True
    storage._connection = MagicMock()
    return storage


class TestRetrieve:
    def test_query_uses_cosine_distance_operator(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=[
            (1, "AAPL-123", "mdna", "some real prose", 0.12),
        ])
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        results = retrieve([0.1, 0.2, 0.3], top_k=5, storage=storage)

        sql = mock_cursor.execute.call_args[0][0]
        assert "<=>" in sql
        assert "ORDER BY" in sql
        assert "LIMIT" in sql
        assert len(results) == 1
        assert results[0]["filing_id"] == "AAPL-123"
        assert results[0]["content"] == "some real prose"

    def test_passes_embedding_and_top_k_as_params(self):
        storage = _make_storage()
        mock_cursor = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=[])
        storage._connection.cursor = MagicMock(return_value=mock_cursor)

        retrieve([0.5, 0.6], top_k=3, storage=storage)

        params = mock_cursor.execute.call_args[0][1]
        assert params == ([0.5, 0.6], 3)

    def test_unavailable_storage_returns_empty(self):
        storage = NormalizedStorage("localhost", 5432, "user", "pass", "db")
        storage._available = False
        assert retrieve([0.1], top_k=5, storage=storage) == []


class TestGenerateAnswer:
    def test_prompt_contains_retrieved_content_and_question(self):
        captured_prompts = []

        def fake_generate(prompt: str) -> tuple[str, float]:
            captured_prompts.append(prompt)
            return "fake answer", 0.0

        retrieved = [
            {"section_id": 1, "filing_id": "AAPL-123", "section_type": "mdna", "content": "Apple's real revenue commentary"},
        ]
        answer, cited = generate_answer("What was revenue?", retrieved, fake_generate)

        assert answer == "fake answer"
        assert cited == ["AAPL-123"]
        assert "Apple's real revenue commentary" in captured_prompts[0]
        assert "What was revenue?" in captured_prompts[0]

    def test_cited_filing_ids_are_deduplicated_and_ordered(self):
        retrieved = [
            {"section_id": 1, "filing_id": "AAPL-123", "section_type": "mdna", "content": "a"},
            {"section_id": 2, "filing_id": "MSFT-456", "section_type": "risk_factors", "content": "b"},
            {"section_id": 3, "filing_id": "AAPL-123", "section_type": "risk_factors", "content": "c"},
        ]
        _, cited = generate_answer("q", retrieved, lambda p: ("a", 0.0))
        assert cited == ["AAPL-123", "MSFT-456"]

    def test_no_retrieved_sections_short_circuits_without_calling_generate_fn(self):
        calls = []
        answer, cited = generate_answer("q", [], lambda p: calls.append(p) or ("x", 0.0))
        assert calls == []
        assert cited == []
        assert "No relevant" in answer


class TestEmbedderLazyLoad:
    def test_model_not_loaded_until_first_embed_call(self):
        """Must not import sentence-transformers or load a model at
        construction time -- only sentence-transformers>=2.2.2 being listed
        in requirements.txt as a demo-only dependency stays true if nothing
        eagerly imports it."""
        embedder = Embedder()
        assert embedder._model is None
