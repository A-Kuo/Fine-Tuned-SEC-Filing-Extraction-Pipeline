"""Local, no-API-key text embeddings for the lightweight RAG demo.

Explicitly scoped as a capability demonstration over real filing data, not
a production extraction path -- the fine-tuned model remains the primary
extraction mechanism. sentence-transformers is a demo-only dependency (see
requirements.txt's comment) and is never imported at module load time, only
on first real use, matching the lazy-load pattern already established by
ExtractionEngine.initialize() (src/extraction/inference.py) and
FinancialLLM.from_config() (src/extraction/model.py) -- so importing this
module, or booting the API server, never pulls in torch/sentence-transformers
unless RAG is actually used.
"""

from __future__ import annotations

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # must match db/migrations/0007_rag_embeddings.sql's vector(384)


class Embedder:
    """Wraps a sentence-transformers model, loaded lazily on first use."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or DEFAULT_MODEL_NAME
        self._model = None

    def initialize(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Loads the model on first call."""
        self.initialize()
        return self._model.encode(texts, convert_to_numpy=True).tolist()
