"""Tests for scripts/parse_xbrl.py's inline-XBRL fact extraction.

No tests existed for this module before -- it was genuinely untested,
matching its own "best-effort... not a full XBRL processor" framing. These
were added while fixing a real bug found during the public.* schema
backfill: the parser ignored the scale/sign attributes inline XBRL uses to
convey units, so a real MSFT filing's revenue tag displaying "137.7" with
scale="9" (billions) was stored as literally 137.7 instead of ~137.7
billion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from parse_xbrl import extract_xbrl_facts, map_to_training_fields


class TestScaleSignHandling:
    def test_ix_nonfraction_applies_scale(self):
        """scale="9" means the displayed digits are in billions."""
        html = '<ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax" scale="9" decimals="-9">137.7</ix:nonFraction>'
        facts = extract_xbrl_facts(html)
        assert facts["ix:RevenueFromContractWithCustomerExcludingAssessedTax"]["value"] == 137.7e9

    def test_ix_nonfraction_no_scale_is_literal(self):
        """No scale attribute -- value is taken as displayed, unchanged."""
        html = '<ix:nonFraction name="us-gaap:EarningsPerShareBasic">5.23</ix:nonFraction>'
        facts = extract_xbrl_facts(html)
        assert facts["ix:EarningsPerShareBasic"]["value"] == 5.23

    def test_ix_nonfraction_applies_negative_sign(self):
        """sign="-" means a contra/negative value displayed as positive digits."""
        html = '<ix:nonFraction name="us-gaap:NetIncomeLoss" sign="-" scale="6">500</ix:nonFraction>'
        facts = extract_xbrl_facts(html)
        assert facts["ix:NetIncomeLoss"]["value"] == -500e6

    def test_xml_path_applies_scale(self):
        """The ElementTree-based fallback path must apply scale identically."""
        xml = '<root><nonFraction name="us-gaap:Assets" scale="6">245122</nonFraction></root>'
        facts = extract_xbrl_facts(xml)
        assert facts["xml:Assets"]["value"] == 245122e6

    def test_malformed_scale_falls_back_to_literal(self):
        """A non-integer scale attribute must not crash extraction."""
        html = '<ix:nonFraction name="us-gaap:Assets" scale="not-a-number">100</ix:nonFraction>'
        facts = extract_xbrl_facts(html)
        assert facts["ix:Assets"]["value"] == 100.0


class TestMapToTrainingFields:
    def test_magnitude_fields_converted_to_millions(self):
        """revenue/net_income/total_assets/total_liabilities are divided by
        1e6 to match MODEL_CARD.md's documented 'millions USD' field
        contract, since extract_xbrl_facts now returns raw USD."""
        facts = {
            "ix:Revenues": {"value": 383_285_000_000.0, "source": "ix_nonFraction"},
            "ix:NetIncomeLoss": {"value": 96_995_000_000.0, "source": "ix_nonFraction"},
        }
        mapped = map_to_training_fields(facts)
        assert mapped["revenue"] == 383_285.0
        assert mapped["net_income"] == 96_995.0

    def test_eps_not_converted_to_millions(self):
        """eps is a per-share dollar figure, not a magnitude -- must not be divided."""
        facts = {"ix:EarningsPerShareBasic": {"value": 6.08, "source": "ix_nonFraction"}}
        mapped = map_to_training_fields(facts)
        assert mapped["eps"] == 6.08

    def test_missing_fact_stays_none(self):
        mapped = map_to_training_fields({})
        assert mapped["revenue"] is None
        assert mapped["eps"] is None
