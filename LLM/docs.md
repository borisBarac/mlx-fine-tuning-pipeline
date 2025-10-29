# F1 Data JSON Format Specification

Dataset:
```python
import polars as pl

# Login using e.g. `huggingface-cli login` to access this dataset
df = pl.read_parquet('hf://datasets/vibingshu/2024_formula1_championship_dataset/data/train-00000-of-00001.parquet')
```

This document describes the JSON format used for the F1 data files in the pipeline.

## JSON Schema

The expected format for each JSON object is:

```json
{
  "prompt": "string",
  "completion": "string"
}
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `prompt` | string | The input text or question for the model |
| `completion` | string | The expected output or answer from the model |

## Example

```json
{
  "prompt": "What is the capital of France?",
  "completion": "Paris."
}
```
