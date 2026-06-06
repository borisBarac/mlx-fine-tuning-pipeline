# PRD: Migrate Fine-Tuning from MLX to Unsloth

## Problem Statement

The pipeline currently depends on `mlx-lm` for LoRA fine-tuning, locking users into a single Apple-Silicon-only backend. Users who want to train on NVIDIA GPUs, or who want access to Unsloth's performance optimizations and broader model support on Apple Silicon, cannot do so without rewriting the training module. The training configuration is also hardcoded and inflexible — model, learning rate, LoRA rank, and chat template are not configurable at the pipeline level.

## Solution

Replace the `mlx-lm` training backend with `unsloth`, which supports both MLX (Apple Silicon) and CUDA (NVIDIA GPU) backends. Expose training configuration as pipeline parameters. Switch the data format from legacy `prompt/completion` JSONL to the standard HuggingFace chat format (`messages`). Add an optional model export step. Rename the training module and pipeline entry point to be backend-agnostic.

## User Stories

1. As a pipeline user, I want to fine-tune models on Apple Silicon using Unsloth's MLX backend, so that I get better performance and memory usage than raw mlx-lm.
2. As a pipeline user, I want to fine-tune models on an NVIDIA GPU using Unsloth's CUDA backend, so that I can use cloud or on-prem GPU hardware.
3. As a pipeline user, I want to select my backend via a `--backend` flag, so that I explicitly control which hardware target is used.
4. As a pipeline user, I want to specify any HuggingFace model via a `--model` flag, so that I can experiment with different base models.
5. As a pipeline user, I want to specify the chat template via a `--chat-template` flag, so that the correct tokenizer template is applied for my model.
6. As a pipeline user, I want to tune `--learning-rate`, `--lora-rank`, and `--max-seq-length`, so that I can control training quality and resource usage.
7. As a pipeline user, I want adapters saved to backend-specific directories (`adapters/mlx/` or `adapters/cuda/`), so that I don't mix incompatible adapter formats.
8. As a pipeline user, I want to optionally export a merged model as safetensors via `--export-format safetensors`, so that I can deploy the fine-tuned model for inference.
9. As a pipeline user, I want training parameter defaults to match Unsloth's recommendations, so that I get good results out of the box.
10. As a pipeline user, I want LoRA configuration to be unified across backends, so that the same parameter values produce consistent behavior regardless of backend.
11. As a pipeline developer, I want the training module named `src/training/` instead of `src/mlx/`, so that the name reflects its backend-agnostic nature.
12. As a pipeline developer, I want the pipeline entry point renamed from `pipeline_ml.py` to `pipeline.py`, so that it doesn't imply MLX-specificity.
13. As a pipeline developer, I want data prep to output `{"messages": [...]}` format, so that it works natively with Unsloth's chat template handling.
14. As a pipeline developer, I want existing data prep tests updated for the new format, so that the test suite stays green after migration.
15. As a pipeline developer, I want the `mlx-lm` dependency replaced with `unsloth`, so that the dependency tree is clean and conflict-free.

## Implementation Decisions

- **Training module**: Rename `src/mlx/train.py` to `src/training/trainer.py`. The module dispatches to Unsloth's API based on the `--backend` parameter. A single file is sufficient because Unsloth provides a unified API for both backends.
- **Pipeline entry point**: Rename `src/pipeline_ml.py` to `src/pipeline.py`. Update the Metaflow `FlowSpec` class with new parameters.
- **New pipeline parameters**: `--backend` (default `"mlx"`), `--model` (default `"unsloth/granite-4.0-1b"`), `--chat-template` (default `"llama-3.1"`), `--learning-rate` (default `2e-4`), `--lora-rank` (default `64`), `--max-seq-length` (default `2048`), `--export-format` (default `none`, options: `none`, `safetensors`).
- **Chat template resolution**: Use `unsloth.chat_templates.get_chat_template(tokenizer, chat_template=<value>)` in the training module. The `--chat-template` value is passed through directly — no auto-mapping or inference from the model name.
- **Data format change**: `transform_parquet_to_jsonl` will output `{"messages": [{"role": "user", "content": <instruction>}, {"role": "assistant", "content": <output>}]}` instead of `{"prompt": <instruction>, "completion": <output>}`.
- **Adapter output**: `adapters/mlx/` when `--backend mlx`, `adapters/cuda/` when `--backend cuda`. Controlled by the training module based on the backend parameter.
- **Export step**: A new optional pipeline step after training. Triggered when `--export-format` is not `none`. Produces a merged safetensors model in the adapter output directory.
- **Dependency change**: Remove `mlx-lm==0.28.3` from `pyproject.toml`, add `unsloth`. Unsloth handles MLX installation on Apple Silicon internally.
- **Default parameter values**: Use Unsloth's recommended defaults (LR `2e-4`, rank `64`, seq length `2048`) rather than MLX's current defaults (LR `1e-5`, rank `8`).

## Testing Decisions

- **What makes a good test**: Tests should verify external behavior (output format, file contents, parameter passing) not implementation details (which backend function is called).
- **Data prep tests**: Update `test_transform.py` to assert the new `{"messages": [...]}` output format instead of `{"prompt": ..., "completion": ...}`. This is the highest-value test seam — it catches regressions in the data format change.
- **Existing test seams**: `test_create_training_data.py`, `test_load.py`, `test_utils.py` should continue passing with minor updates to reflect the new output format.
- **Training tests**: Skipped for now. The training module will be in flux during migration. Add integration tests once the API stabilizes.
- **No new test seams introduced**: All testing happens through existing test files.

## Out of Scope

- MLX-format adapter export (GGUF, MLX-specific quantization formats)
- Multi-turn conversation support in data prep
- Hyperparameter tuning or automated parameter search
- Model evaluation / benchmarking step
- Cloud deployment or remote training orchestration
- Training module unit tests (deferred until API stabilizes)
- Auto-detection of chat template from model metadata
- Per-backend LoRA default values

## Further Notes

- This is a clean-break migration — old files are renamed in-place, not kept alongside new ones.
- The pipeline's Metaflow orchestration, parallel data processing, and Polars-based data handling remain unchanged.
- The current Granite model (`mlx-community/granite-4.0-1b-base-4bit`) will be replaced by its Unsloth equivalent as the default.
- Unsloth's `get_chat_template` function handles template validation internally, reducing the need for custom error handling.
