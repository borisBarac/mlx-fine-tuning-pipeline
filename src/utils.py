import re


def is_valid_hf_parquet_link(url: str) -> bool:
    """Check if string is a valid Hugging Face model link pointing to a parquet file."""
    if not isinstance(url, str):
        return False

    # Pattern: hf://datasets/username/dataset_name/path/to/file.parquet
    pattern = r"^hf://datasets/[^/]+/[^/]+/.*\.parquet$"
    return bool(re.match(pattern, url))
