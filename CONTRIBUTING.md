# Contributing to FinDocAnalyzer

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Ways to Contribute

- **Report bugs**: Open an issue with a minimal reproduction and environment details
- **Suggest enhancements**: Open an issue describing the feature and its use case
- **Submit code**: Follow the workflow below for pull requests
- **Improve documentation**: Fix typos, clarify explanations, add examples
- **Add test cases**: Increase coverage, especially for edge cases in parsing

## Development Setup

```bash
git clone https://github.com/akuo6/financial-llm.git
cd financial-llm

pip install -r requirements.txt

# Run tests to verify setup
make test
```

## Code Style

- **Formatter**: `ruff format` (line length 100)
- **Linter**: `ruff check`
- **Type checker**: `mypy src/ --ignore-missing-imports`
- **Docstrings**: Google style for public modules

Run checks before committing:
```bash
make lint
make format
make typecheck
make test
```

## Pull Request Workflow

1. **Fork** the repository and create a branch from `main`
2. **Make changes** with clear, focused commits
3. **Add tests** for new functionality
4. **Run the full test suite** — all 103 tests must pass
5. **Update documentation** if changing user-facing behavior
6. **Submit PR** with a clear description of changes and motivation

### PR Checklist

- [ ] Tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] No new lint errors (`make lint`)
- [ ] Docstrings updated for public APIs
- [ ] README updated if user-facing changes
- [ ] CLAUDE.md updated if architecture changes

## Project Structure Conventions

- `src/` — Core library (model loading, inference, postprocessing, database)
- `training/` — Fine-tuning pipeline (train, callbacks, data collator)
- `serving/` — Production serving (FastAPI, vLLM, batch inference)
- `evaluation/` — Accuracy and performance measurement
- `monitoring/` — Production observability
- `scripts/` — Data preparation and setup utilities
- `tests/` — Test suite (CPU-only, no GPU required)
- `notebooks/` — Colab notebooks for GPU workloads

## Testing Guidelines

- All tests must run without a GPU (mock GPU-dependent code)
- Use `pytest` fixtures for common setups
- Target 80%+ coverage for new modules
- Edge cases for JSON parsing are especially important

## Commit Message Style

Use clear, descriptive commit messages:

```
feat: add EDGAR data fetcher for real SEC filings

fix: handle truncated JSON with missing closing brace

docs: update API endpoint documentation

test: add validation tests for null financial fields
```

## Questions?

Open a discussion on GitHub or refer to `.cursor/CLAUDE.md` for architectural context (agent guidance for continued development).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
