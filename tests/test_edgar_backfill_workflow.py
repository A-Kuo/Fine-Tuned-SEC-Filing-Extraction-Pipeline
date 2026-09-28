"""Pins the safety properties of .github/workflows/backfill_edgar_filings.yml.

This workflow writes real rows to production Supabase and calls SEC's live
API for every company in scope, so it must never fire on an unrelated push,
must fail fast (not silently no-op) if the schema migration hasn't been
applied yet, and its dry-run input must actually reach the script.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "backfill_edgar_filings.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _on(workflow):
    # PyYAML parses the bare key `on` as boolean True.
    return workflow.get("on") or workflow.get(True)


def _steps(workflow):
    return workflow["jobs"]["backfill"]["steps"]


def _step(workflow, fragment):
    return next(s for s in _steps(workflow) if fragment in (s.get("name") or ""))


class TestTrigger:
    def test_the_only_trigger_is_a_manual_dispatch(self, workflow):
        """Unlike backfill_live_schema.yml, this must not run on push -- it
        hits real SEC infrastructure and grows the database every time."""
        assert list(_on(workflow)) == ["workflow_dispatch"]

    def test_dispatch_inputs_support_a_small_bounded_proof_run(self, workflow):
        inputs = _on(workflow)["workflow_dispatch"]["inputs"]
        assert inputs["limit_companies"]["default"] == "50"
        assert inputs["dry_run"]["type"] == "boolean"
        assert inputs["dry_run"]["default"] is False


class TestSchemaGuard:
    def test_fails_fast_if_the_edgar_schema_is_missing(self, workflow):
        step = _step(workflow, "edgar schema exists")
        assert "to_regclass('edgar.filings')" in step["run"]
        assert "sys.exit(1)" in step["run"]

    def test_schema_check_runs_before_the_backfill_step(self, workflow):
        steps = _steps(workflow)
        names = [s.get("name") or "" for s in steps]
        check_idx = next(i for i, n in enumerate(names) if "edgar schema exists" in n)
        backfill_idx = next(i for i, n in enumerate(names) if "Backfill edgar.entities" in n)
        assert check_idx < backfill_idx


class TestInputsReachTheScript:
    def test_dry_run_input_is_forwarded_as_a_flag(self, workflow):
        step = _step(workflow, "Backfill edgar.entities")
        assert "--dry-run" in step["run"]
        assert "inputs.dry_run" in step["run"]

    def test_limit_companies_and_explicit_tickers_are_both_wired_in(self, workflow):
        step = _step(workflow, "Backfill edgar.entities")
        assert "--limit-companies" in step["run"]
        assert "inputs.limit_companies" in step["run"]
        assert "--tickers" in step["run"]
        assert "inputs.tickers" in step["run"]

    def test_verify_step_is_skipped_on_a_dry_run(self, workflow):
        step = _step(workflow, "Verify")
        assert step["if"] == "${{ inputs.dry_run != 'true' }}"


class TestCredentials:
    def test_credentials_come_from_the_same_secrets_as_the_other_supabase_workflow(self, workflow):
        env = workflow["jobs"]["backfill"]["env"]
        assert env["POSTGRES_HOST"] == "${{ secrets.SUPABASE_DB_HOST }}"
        assert env["POSTGRES_PASSWORD"] == "${{ secrets.SUPABASE_DB_PASSWORD }}"

    def test_missing_required_secrets_fail_the_run(self, workflow):
        step = _step(workflow, "Check required secrets")
        assert "SUPABASE_DB_HOST secret is not set" in step["run"]
        assert "exit 1" in step["run"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
