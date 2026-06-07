# AGENTS.md

## Project

Python pipeline for fine-tuning language models on Apple Silicon and NVIDIA GPUs. Uses Metaflow for orchestration, Unsloth/PyTorch for training (CUDA + MPS), and Polars for data processing.

**This is a Python project.** Node.js/Bun is only used by the Sandcastle agent runner in `.sandcastle/`.

## Structure

- `src/data_prep/` — document loading, transformation, training data creation
- `src/training/` — model training logic
- `src/pipeline.py` — combined Metaflow pipeline (data prep + training)
- `src/data_prep_flow.py` — standalone data preparation pipeline
- `src/training_flow.py` — standalone training pipeline
- `src/utils.py` — shared utilities
- `test/` — pytest tests
- `PRD/` — product requirement documents

## Commands

| Task | Command |
|---|---|
| Install dependencies | `uv sync` |
| Run tests | `uv run pytest` |
| Run tests with coverage | `uv run pytest --cov` |
| Lint | `uvx ruff check` |
| Format | `uvx ruff format` |
| Lint + format | `./lint.sh` |

## Requirements

- Python 3.12
- uv package manager
- Apple Silicon (M-series) or NVIDIA GPU for training

## E2E Tests

**Do not run e2e tests without explicit user approval.** These tests:
- Download large models (~5-6GB)
- Require Apple Silicon GPU
- Take several minutes to complete
