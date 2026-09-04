"""Import-smoke test: every module under src/, serving/, monitoring/,
evaluation/, and db/sync/ must import cleanly.

Cheap and would have caught, mechanically, the exact bug this session found
and fixed by hand: the src/ reorg (flat -> src/core/, src/extraction/,
src/storage/) silently broke both Kaggle notebooks' imports (src.model,
src.inference, src.config) with no test anywhere to catch it, since nothing
imported those modules in CI.

Excludes modules that import the torch/peft/trl stack directly
(src/extraction/model.py, training/train.py, training/data_collator.py) --
those genuinely need packages requirements-ci.txt deliberately omits ("no
torch/vLLM/GPU", keeps CI at ~1-2 min). training/callbacks.py and
training/ablation_config.py ARE included: the former only needs
`transformers` (torch-free base install, added to requirements-ci.txt this
session), the latter is dependency-free by design.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent

# Needs torch/peft/trl -- not in requirements-ci.txt by design, see module docstring.
EXCLUDED_MODULES = {
    "src.extraction.model",
    "training.train",
    "training.data_collator",
}


def _discover_modules(package_dir: str) -> list[str]:
    modules = []
    for path in sorted((REPO_ROOT / package_dir.replace(".", "/")).rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        rel = path.relative_to(REPO_ROOT).with_suffix("")
        dotted = ".".join(rel.parts)
        if dotted not in EXCLUDED_MODULES:
            modules.append(dotted)
    return modules


ALL_MODULES = (
    _discover_modules("src")
    + _discover_modules("evaluation")
    + _discover_modules("monitoring")
    + _discover_modules("serving")
    + _discover_modules("db/sync")
    + ["training.callbacks", "training.ablation_config"]
)


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports_cleanly(module_name):
    importlib.import_module(module_name)


def test_at_least_the_known_modules_were_discovered():
    """Guards against _discover_modules() silently finding zero files (e.g.
    a path typo) and this whole test file passing vacuously."""
    assert len(ALL_MODULES) >= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
