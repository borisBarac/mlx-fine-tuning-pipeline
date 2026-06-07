# AGENTS.md

## Project

Python pipeline for fine-tuning language models on Apple Silicon. Uses Metaflow for orchestration, MLX/Unsloth for training, and Polars for data processing.

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
- Apple Silicon (M-series) for MLX training
