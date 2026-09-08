-- Lightweight RAG demo: persist section prose (previously discarded before
-- it ever reached this table -- see src/extraction/normalizer.py) and add
-- an embedding column for cosine-similarity retrieval over it.
--
-- vector(384) matches sentence-transformers/all-MiniLM-L6-v2 (local, no API
-- key, no cost -- appropriate for a capability demo, not a production
-- extraction path). No ivfflat/hnsw index: real data scale here is 5-6
-- filings and a few dozen sections, where a brute-force `<=>` cosine scan
-- is entirely adequate; an ANN index would be speculative infrastructure
-- for data volume that doesn't justify it.

alter table intel.filing_sections
    add column if not exists content text,
    add column if not exists embedding vector(384);
