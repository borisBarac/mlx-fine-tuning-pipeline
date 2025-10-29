from pathlib import Path

import polars as pl
from polars import LazyFrame


def download_parquet_to_cache(
    dataset_path: str, cache_dir: str = "LLM/parquet_sets", override: bool = True
) -> str:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Extract filename from the path
    filename = Path(dataset_path).name
    file_path = cache_path / filename

    try:
        # Polars can directly read Hugging Face datasets using hf:// protocol
        # This handles authentication and downloading automatically
        print(f"Loading parquet file from: {dataset_path}")

        # Load dataset using polars with hf:// protocol support
        lazy_frame: LazyFrame = pl.scan_parquet(dataset_path)

        # Convert to eager frame and save to cache
        df = lazy_frame.collect()

        # Override existing file if it exists and override=True
        if override and file_path.exists():
            print(f"Overwriting existing file: {file_path}")

        df.write_parquet(file_path)

        return str(file_path)

    except Exception as e:
        raise Exception(
            f"Failed to download and cache parquet file '{dataset_path}': {str(e)}"
        )
