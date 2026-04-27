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
        if file_path.exists() and not override:
            print(f"Using cached file: {file_path}")
            return str(file_path)

        print(f"Loading parquet file from: {dataset_path}")

        lazy_frame: LazyFrame = pl.scan_parquet(dataset_path)
        df = lazy_frame.collect()

        if override and file_path.exists():
            print(f"Overwriting existing file: {file_path}")

        df.write_parquet(file_path)

        return str(file_path)

    except Exception as e:
        raise Exception(
            f"Failed to download and cache parquet file '{dataset_path}': {str(e)}"
        )
