# ML Pipeline Tutorial: Fine-tune Models

Learn to build a complete ML pipeline for fine-tuning models using F1 racing data. This tutorial covers data processing, workflow management, and model training.

## What You'll Build

A functional ML pipeline that:
- Loads datasets from multiple sources
- Processes data in parallel
- Splits data for training/validation
- Fine-tunes models with MLX
- Manages workflows with Metaflow

#### Example Pipeline Run
To see what a complete pipeline execution looks like, check out the [pipeline output example](./pipline_output_example.md). This shows a real run with:


## Prerequisites

Install these tools first:

- **Python 3.12+** - Programming environment
- **UV** - Fast package installer ([install](https://docs.astral.sh/uv/))
- **Docker** - For Metaflow services ([install](https://www.docker.com/))

## Step 1: Setup Environment

### Important: Set up API Key
Metaflow does now work with local env files, just AWS and Azure secrets, so for local run we are stuck with this.

Before running the pipeline, you need to set up your Hugging Face API key. Replace `your HF token goes here` in the `login(token="your HF token goes here")` function with your actual Hugging Face token. You can find your token in your Hugging Face account settings.

### Install dependencies:

```bash
uv pip install -r pylock.toml --preview-features pylock
```

This installs:
- **Metaflow** - Workflow orchestration
- **MLX** - Apple's ML framework
- **Polars** - Fast data processing
- **Other ML utilities**

## Step 2: Start Metaflow Services

```bash
uv run metaflow-dev up
```

This starts background services for tracking and storage.

## Step 3: Pipeline Overview

The pipeline has 4 main stages:

1. **Data Loading** - Loads local or remote datasets
2. **Parallel Processing** - Splits data into chunks for speed
3. **Data Splitting** - Creates train/validation sets (95%/5%)
4. **Model Training** - Fine-tunes models with MLX

## Step 4: Run the Pipeline

### Using Local Data
```bash
uv run python ./src/pipeline_ml.py run --parquet-path ./LLM/parquet_sets/train-00000-of-00001.parquet
```

### Using Hugging Face Data
```bash
uv run python ./src/pipeline_ml.py run --parquet-path hf://datasets/vibingshu/2024_formula1_championship_dataset/data/train-00000-of-00001.parquet
```

## Step 5: Customize Parameters

```bash
uv run python ./src/pipeline_ml.py run \
  --parquet-path ./LLM/parquet_sets/train-00000-of-00001.parquet \
  --chunk-size 500 \
  --training-iters 200 \
  --batch-size 8
```

**Parameters:**
- `--chunk-size`: Rows per chunk (default: 1000)
- `--training-iters`: Training iterations (default: 100)
- `--batch-size`: Training batch size (default: 4)

## Technology Stack

### MLX
- Apple's ML framework for M1/M2/M3 chips
- Fast training with GPU acceleration
- Memory efficient

### Metaflow
- Manages complex ML workflows
- Tracks experiments and results
- Handles errors and resources

### Polars
- Faster than pandas for big data
- Parallel processing
- Memory efficient

## Pipeline Structure

### Key Files
- `src/data_prep/` - Data loading and transformation
- `src/mlx/` - Model training code
- `src/pipeline_ml.py` - Main pipeline orchestration

## Step 5: Data Preparation

Data is prepared in two steps:

1. **Parallel Processing** - Data is split into chunks and processed in parallel across multiple machines.
```python
# Split data into chunks
chunks = self.create_chunks(data, chunk_size)
# Process chunks in parallel
results = self.parallel_map(self.process_chunk, chunks)
```

2. **Result Merging** - The results from the previous step are merged to produce the final dataset.
```python
temp_combined = Path(self.temp_dir) / "combined.jsonl"
merge_result = merge_jsonl_files(self.chunk_files, str(temp_combined))
```

## Step 6: Monitor Pipeline

### Check Status
```bash
uv run metaflow status
```

### View Logs
```bash
uv run metaflow logs <run-id>
```

### UI Dashboard
Visit `http://localhost:8080` for visual monitoring.

## Step 7: Test Pipeline

```bash
uv run pytest test/
```

## Next Steps

### Possible Additions
- Model evaluation metrics
- Hyperparameter tuning
- Model deployment
- Data versioning
- Experiment tracking

### Production Tips
- Cloud deployment (AWS, GCP, Azure)
- Performance monitoring
- Security controls
- Automated testing

## Troubleshooting

### Memory Issues
- Reduce `--chunk-size`
- Close other apps
- Check system resources

### Metaflow Problems
- Restart: `uv run metaflow-dev restart`
- Check Docker: `docker ps`
- Reset: `uv run metaflow-dev reset`

### Training Failures
- Check data format
- Verify model compatibility
- Reduce batch size

## Conclusion

You've built a complete ML pipeline! This pipeline shows:

✅ **Data Engineering** - Efficient data processing
✅ **Workflow Management** - Robust orchestration
✅ **Machine Learning** - Model fine-tuning
✅ **Scalability** - Parallel processing
✅ **Reproducibility** - Trackable experiments

Apply these skills to other ML projects:
- NLP tasks
- Computer vision
- Time series
- Recommendation systems

Happy building! 🚀