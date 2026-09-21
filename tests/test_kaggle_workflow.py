"""Pins the safety properties of .github/workflows/kaggle_training.yml.

A push to main from a workflow, or an expensive GPU run, must never happen by
accident, and a failed run must not throw its log away. These are the
properties that are easy to break with a well-meaning edit.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "kaggle_training.yml"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _inputs(workflow):
    # PyYAML parses the bare key `on` as boolean True.
    return (workflow.get("on") or workflow.get(True))["workflow_dispatch"]["inputs"]


def _steps(workflow):
    return workflow["jobs"]["train-on-kaggle"]["steps"]


def _step(workflow, fragment):
    return next(s for s in _steps(workflow) if fragment in (s.get("name") or ""))


class TestInputs:
    def test_runs_default_to_the_cheap_smoke_mode(self, workflow):
        mode = _inputs(workflow)["mode"]
        assert mode["type"] == "choice"
        assert mode["options"] == ["smoke", "full"]
        assert mode["default"] == "smoke"

    def test_committing_results_is_off_by_default(self, workflow):
        commit = _inputs(workflow)["commit_results"]
        assert commit["type"] == "boolean"
        assert commit["default"] is False

    def test_the_only_trigger_is_a_manual_dispatch(self, workflow):
        assert list((workflow.get("on") or workflow.get(True))) == ["workflow_dispatch"]


class TestPushToMainIsOptIn:
    def test_the_commit_step_needs_success_a_full_run_and_the_opt_in(self, workflow):
        condition = _step(workflow, "Commit results")["if"]
        assert "success()" in condition
        assert "inputs.mode == 'full'" in condition
        assert "inputs.commit_results" in condition

    def test_no_other_step_pushes(self, workflow):
        for step in _steps(workflow):
            if "Commit results" in (step.get("name") or ""):
                continue
            assert "git push" not in (step.get("run") or ""), step.get("name")


class TestRunPlumbing:
    def test_the_mode_reaches_the_script_through_an_env_var_not_inline_interpolation(self, workflow):
        step = _step(workflow, "Submit training job")
        assert step["env"]["RUN_MODE"] == "${{ inputs.mode }}"
        assert '--mode "$RUN_MODE"' in step["run"]
        assert "${{" not in step["run"]

    def test_a_failed_run_still_downloads_its_kernel_log(self, workflow):
        step = _step(workflow, "kernel log")
        assert step["if"] == "failure()"
        assert "--logs-only" in step["run"]

    def test_the_artifact_always_uploads_and_includes_the_results_folder(self, workflow):
        step = _step(workflow, "Upload run output")
        assert step["if"] == "always()"
        assert ".kaggle_output/" in step["with"]["path"]
        assert "notebooks/results/" in step["with"]["path"]

    def test_kaggle_credentials_come_from_secrets(self, workflow):
        env = _step(workflow, "Submit training job")["env"]
        assert env["KAGGLE_USERNAME"] == "${{ secrets.KAGGLE_USERNAME }}"
        assert env["KAGGLE_KEY"] == "${{ secrets.KAGGLE_KEY }}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
