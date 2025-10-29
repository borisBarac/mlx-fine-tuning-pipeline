from src.utils import is_valid_hf_parquet_link


def test_valid_hf_parquet_links():
    """Test valid Hugging Face parquet links."""
    valid_links = [
        "hf://datasets/vibingshu/2024_formula1_championship_dataset/data/train-00000-of-00001.parquet",
        "hf://datasets/username/dataset_name/file.parquet",
        "hf://datasets/user-name/dataset-name/path/to/file.parquet",
        "hf://datasets/user123/dataset_456/data/train.parquet",
    ]

    for link in valid_links:
        assert is_valid_hf_parquet_link(link), f"Should be valid: {link}"


def test_invalid_hf_parquet_links():
    """Test invalid Hugging Face parquet links."""
    invalid_links = [
        "https://huggingface.co/datasets/username/dataset/file.parquet",
        "hf://datasets/username/dataset/file.csv",
        "hf://datasets/username/file.parquet",
        "hf://models/username/model/file.parquet",
        "hf://datasets//dataset/file.parquet",
        "hf://datasets/username//file.parquet",
        "not_a_url",
        "",
        None,
        123,
        "hf://datasets/username/dataset/file.txt",
        "hf://datasets/username/dataset/",
    ]

    for link in invalid_links:
        assert not is_valid_hf_parquet_link(link), f"Should be invalid: {link}"


def test_edge_cases():
    """Test edge cases."""
    # Empty string
    assert not is_valid_hf_parquet_link("")

    # Non-string input
    assert not is_valid_hf_parquet_link(None)  # type: ignore
    assert not is_valid_hf_parquet_link(123)  # type: ignore
    assert not is_valid_hf_parquet_link([])  # type: ignore

    # Missing parquet extension
    assert not is_valid_hf_parquet_link("hf://datasets/user/dataset/file")

    # Wrong extension
    assert not is_valid_hf_parquet_link("hf://datasets/user/dataset/file.csv")

    # Multiple dots in filename but not parquet
    assert not is_valid_hf_parquet_link("hf://datasets/user/dataset/file.txt.backup")

    # Valid with multiple dots but parquet at end
    assert is_valid_hf_parquet_link("hf://datasets/user/dataset/file.v2.parquet")
