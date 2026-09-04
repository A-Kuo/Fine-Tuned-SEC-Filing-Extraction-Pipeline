"""Tests for scripts/fetch_edgar.py's pure helper functions.

Regression coverage for two real bugs found and fixed this session:
1. html_to_visible_text() -- the fetcher used to save raw HTML/inline-XBRL
   straight to .txt with zero stripping, which meant section detection was
   being tested against markup soup, not prose.
2. rebuild_manifest() -- manifest.jsonl used to be truncated (mode="w")
   on every non-resume run even though the actual filing files from
   earlier runs were untouched, silently losing their manifest entries.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_edgar import html_to_visible_text, rebuild_manifest


class TestHtmlToVisibleText:
    def test_strips_tags_to_visible_text(self):
        html = "<html><body><p>Hello world</p></body></html>"
        text = html_to_visible_text(html)
        assert "Hello world" in text
        assert "<p>" not in text

    def test_strips_hidden_inline_xbrl_header_block(self):
        """The <ix:header> block sits inside <div style="display:none"> --
        invisible in a browser, but get_text() alone doesn't know CSS."""
        html = (
            '<html><body>'
            '<div style="display:none"><ix:header>'
            '<ix:nonNumeric name="dei:AmendmentFlag">false</ix:nonNumeric>'
            '</ix:header></div>'
            '<p>Visible filing text</p>'
            '</body></html>'
        )
        text = html_to_visible_text(html)
        assert "Visible filing text" in text
        assert "AmendmentFlag" not in text
        assert "false" not in text

    def test_strips_script_and_style_tags(self):
        html = "<html><body><script>alert(1)</script><style>.x{}</style><p>Real text</p></body></html>"
        text = html_to_visible_text(html)
        assert "Real text" in text
        assert "alert" not in text

    def test_collapses_blank_lines_without_losing_paragraphs(self):
        html = "<html><body><p>First</p><p>Second</p></body></html>"
        text = html_to_visible_text(html)
        lines = [l for l in text.splitlines() if l.strip()]
        assert lines == ["First", "Second"]


class TestRebuildManifest:
    def test_rebuilds_from_meta_json_files_on_disk(self, tmp_path):
        meta1 = {"ticker": "AAPL", "accessionNumber": "0001", "form": "10-K"}
        meta2 = {"ticker": "MSFT", "accessionNumber": "0002", "form": "10-K"}
        (tmp_path / "AAPL_0001.meta.json").write_text(json.dumps(meta1))
        (tmp_path / "MSFT_0002.meta.json").write_text(json.dumps(meta2))

        entries = rebuild_manifest(tmp_path)

        tickers = {e["ticker"] for e in entries}
        assert tickers == {"AAPL", "MSFT"}

    def test_survives_a_prior_run_truncating_manifest_jsonl(self, tmp_path):
        """The exact bug scenario: manifest.jsonl is empty/stale, but the
        real .meta.json files from an earlier run are still on disk."""
        meta = {"ticker": "KO", "accessionNumber": "0003", "form": "10-K"}
        (tmp_path / "KO_0003.meta.json").write_text(json.dumps(meta))
        (tmp_path / "manifest.jsonl").write_text("")  # simulates the truncation bug

        entries = rebuild_manifest(tmp_path)
        assert len(entries) == 1
        assert entries[0]["ticker"] == "KO"

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert rebuild_manifest(tmp_path) == []

    def test_skips_unreadable_meta_file_without_crashing(self, tmp_path):
        (tmp_path / "good.meta.json").write_text(json.dumps({"ticker": "GOOD"}))
        (tmp_path / "bad.meta.json").write_text("{not valid json")

        entries = rebuild_manifest(tmp_path)
        assert len(entries) == 1
        assert entries[0]["ticker"] == "GOOD"

    def test_sorted_by_filename_for_deterministic_order(self, tmp_path):
        (tmp_path / "ZZZ_0001.meta.json").write_text(json.dumps({"ticker": "ZZZ"}))
        (tmp_path / "AAA_0001.meta.json").write_text(json.dumps({"ticker": "AAA"}))

        entries = rebuild_manifest(tmp_path)
        assert [e["ticker"] for e in entries] == ["AAA", "ZZZ"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
