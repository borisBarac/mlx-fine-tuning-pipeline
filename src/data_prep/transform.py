import json
from pathlib import Path
from typing import Optional

import polars as pl


def transform_parquet_to_jsonl(
    parquet_path: str,
    output_path: Optional[str] = None,
    row_range: Optional[tuple[int, int]] = None,
) -> str:
    """
    Transform a parquet file with 'instruction' and 'output' columns
    to a JSONL file with 'prompt' and 'completion' fields.

    Args:
        parquet_path: Path to input parquet file
        output_path: Optional output path (auto-generated if not provided)

    Returns:
        Path to the created JSONL file

    Raises:
        FileNotFoundError: If input parquet file doesn't exist
        ValueError: If required columns are missing or data is malformed
    """
    # Validate input file exists
    input_file = Path(parquet_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")

    # Load parquet with polars
    try:
        df = pl.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(f"Failed to read parquet file: {str(e)}")

    # Validate required columns exist
    required_columns = ["instruction", "output"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check if row_range is specified and apply it
    if row_range is not None:
        start_row, end_row = row_range
        if start_row < 0 or end_row > len(df) or start_row >= end_row:
            raise ValueError(
                f"Invalid row range: {row_range}. Must be within 0-{len(df) - 1} and start < end"
            )
        df = df.slice(start_row, end_row - start_row)

    # Filter malformed data (null/empty values)
    df_clean = df.filter(
        (
            pl.col("instruction").is_not_null()
            & (pl.col("instruction").str.len_chars() > 0)
        )
        & (pl.col("output").is_not_null() & (pl.col("output").str.len_chars() > 0))
    )

    # Transform columns: instruction -> prompt, output -> completion
    df_transformed = df_clean.select(
        [pl.col("instruction").alias("prompt"), pl.col("output").alias("completion")]
    )

    # Generate output path if not provided
    if output_path is None:
        input_stem = input_file.stem
        output_file = input_file.parent / f"{input_stem}.jsonl"
    else:
        output_file = Path(output_path)

    # Write as JSONL
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for row in df_transformed.iter_rows(named=True):
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        raise ValueError(f"Failed to write JSONL file: {str(e)}")

    return str(output_file)


def create_training_data(parquet_path: str, output_dir: str) -> tuple[str, str]:
    """
    Create training data from parquet file, splitting into train and validation sets.

    Args:
        parquet_path: Path to input parquet file
        output_dir: Directory where train.jsonl and valid.jsonl will be created

    Returns:
        Tuple of (train_file_path, valid_file_path)

    Raises:
        FileNotFoundError: If input parquet file doesn't exist
        ValueError: If required columns are missing or data is malformed
    """
    # Validate input file exists
    input_file = Path(parquet_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and transform data using existing function
    temp_jsonl = transform_parquet_to_jsonl(parquet_path)

    # Read the transformed data as JSONL
    df = pl.read_ndjson(temp_jsonl)

    # Calculate split point (last 5% for validation)
    total_rows = len(df)
    split_point = int(total_rows * 0.95)

    # Split data
    train_df = df.slice(0, split_point)
    valid_df = df.slice(split_point)

    # Define output file paths
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"

    # Write train.jsonl
    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_df.iter_rows(named=True):
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    # Write valid.jsonl
    with open(valid_file, "w", encoding="utf-8") as f:
        for row in valid_df.iter_rows(named=True):
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    # Clean up temporary file
    Path(temp_jsonl).unlink()

    return str(train_file), str(valid_file)
