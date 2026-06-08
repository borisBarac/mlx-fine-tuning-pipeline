# LoRA Fine-Tuning Pipeline

Fine-tune language models on Apple Silicon (MPS) and NVIDIA GPUs (CUDA) with Metaflow orchestration, Unsloth/PyTorch training, and Polars data processing.

The pipeline converts documents to text, generates synthetic QA pairs with a teacher model, and fine-tunes a student model via LoRA.

## Prerequisites

- **Python 3.12+**
- **uv** — fast package manager ([install](https://docs.astral.sh/uv/))
- **Apple Silicon (M-series)** or **NVIDIA GPU** for training

## Setup

### Install dependencies

```bash
uv sync
```

### Configure API keys

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` or `OPENAI_API_KEY` | Teacher model API access (QA generation) |
| `HF_TOKEN` | Hugging Face access for gated models (optional) |

## Usage

### Full pipeline (data prep + training)

```bash
uv run python ./src/pipeline.py run \
  --docs-path ./my-documents \
  --teacher-model deepseek-chat \
  --num-examples 200 \
  --training-iters 100 \
  --model unsloth/gemma-3-1b-it
```

### Data preparation only

```bash
uv run python ./src/data_prep_flow.py run \
  --docs-path ./my-documents \
  --num-examples 200
```

### Training only

```bash
uv run python ./src/training_flow.py run \
  --data-path ./LLM/data \
  --training-iters 100 \
  --model unsloth/gemma-3-1b-it
```

## Pipeline Stages

1. **Document Conversion** — Converts PDF, DOCX, PPTX, and TXT files to structured text using Docling
2. **QA Generation** — Generates synthetic question-answer pairs from converted text using a teacher model via OpenAI-compatible API
3. **Parallel Processing** — Transforms parquet data into JSONL chunks using Metaflow's `parallel_map`
4. **Dataset Splitting** — Splits data into train/validation sets (adaptive: 5-15% validation)
5. **Model Training** — Fine-tunes a model using Unsloth (CUDA) or mlx-tune (MPS) with LoRA

## Parameters

### Data Preparation

| Parameter | Default | Description |
|---|---|---|
| `--docs-path` | (required) | Directory of documents to convert |
| `--docs-output` | `<docs-path>/converted` | Output directory for converted files |
| `--data-output` | `src/LLM/data` | Output directory for train.jsonl/valid.jsonl |
| `--chunk-size` | `1000` | Rows per chunk for parallel processing |
| `--num-threads` | `4` | Threads per document converter |
| `--teacher-model` | `deepseek-chat` | Model for QA generation |
| `--num-examples` | `200` | Total QA pairs to generate |
| `--generation-chunk-size` | `2000` | Words per chunk fed to teacher model |
| `--api-base` | `https://openrouter.ai/api/v1` | OpenAI-compatible API endpoint |
| `--api-key` | (env var) | API key (falls back to `OPENROUTER_API_KEY` or `OPENAI_API_KEY`) |

### Training

| Parameter | Default | Description |
|---|---|---|
| `--training-iters` | `100` | Number of training steps |
| `--batch-size` | `4` | Training batch size |
| `--backend` | `auto` | Backend: `auto`, `mlx`, or `cuda` |
| `--model` | `unsloth/gemma-3-1b-it` | HuggingFace model identifier |
| `--chat-template` | (none) | Chat template name |
| `--learning-rate` | `2e-4` | Learning rate |
| `--lora-rank` | `64` | LoRA rank |
| `--max-seq-length` | `2048` | Maximum sequence length |
| `--export-format` | `none` | Export format: `none` or `safetensors` |

## Project Structure

```
src/
  pipeline.py            # Combined pipeline (data prep + training)
  data_prep_flow.py      # Standalone data preparation pipeline
  training_flow.py       # Standalone training pipeline
  data_prep/
    convert.py           # Document-to-text conversion (Docling)
    generate.py          # Synthetic QA generation (teacher model)
    transform.py         # Parquet-to-JSONL, chunking, splitting
  training/
    trainer.py           # Training logic (Unsloth + mlx-tune)
test/                    # pytest tests
```

## Testing

```bash
uv run pytest
```

## Linting

```bash
uvx ruff check
uvx ruff format
```

## Technology Stack

- **Metaflow** — Workflow orchestration, parallel processing, resource management
- **Unsloth** — GPU-accelerated LoRA fine-tuning (CUDA)
- **mlx-tune** — Apple Silicon LoRA fine-tuning (MPS)
- **Docling** — Document parsing (PDF, DOCX, PPTX, TXT)
- **OpenAI API** — Teacher model for synthetic QA generation
- **Polars** — Fast data processing
