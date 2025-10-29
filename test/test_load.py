from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from polars import LazyFrame

from src.data_prep.load import download_parquet_to_cache


class TestDownloadParquetToCache:
    """Test cases for download_parquet_to_cache function."""

    @patch("src.data_prep.load.pl.scan_parquet")
    @patch("builtins.open", new_callable=mock_open)
    def test_download_parquet_to_cache_success(self, mock_file, mock_scan):
        """Test successful download and caching of parquet file."""
        # Setup mocks
        mock_lazy_frame = Mock(spec=LazyFrame)
        mock_scan.return_value = mock_lazy_frame
        mock_df = Mock()
        mock_lazy_frame.collect.return_value = mock_df

        # Test parameters
        dataset_path = "hf://datasets/test/data.parquet"
        cache_dir = "test_cache"
        expected_file_path = "test_cache/data.parquet"

        # Execute function
        result = download_parquet_to_cache(dataset_path, cache_dir)

        # Assertions
        assert result == expected_file_path
        mock_scan.assert_called_once_with(dataset_path)
        mock_lazy_frame.collect.assert_called_once()
        mock_df.write_parquet.assert_called_once_with(Path(expected_file_path))

    @patch("src.data_prep.load.pl.scan_parquet")
    def test_download_parquet_to_cache_with_default_cache_dir(self, mock_scan):
        """Test download with default cache directory."""
        mock_lazy_frame = Mock(spec=LazyFrame)
        mock_scan.return_value = mock_lazy_frame
        mock_df = Mock()
        mock_lazy_frame.collect.return_value = mock_df

        dataset_path = "hf://datasets/test/data.parquet"

        with patch("src.data_prep.load.Path.mkdir") as mock_mkdir:
            result = download_parquet_to_cache(dataset_path)

            assert result == "LLM/parquet_sets/data.parquet"
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("src.data_prep.load.pl.scan_parquet")
    def test_download_parquet_to_cache_exception_handling(self, mock_scan):
        """Test exception handling when download fails."""
        mock_scan.side_effect = Exception("Network error")

        dataset_path = "hf://datasets/test/data.parquet"

        with pytest.raises(Exception) as exc_info:
            download_parquet_to_cache(dataset_path)

        assert "Failed to download and cache parquet file" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)

    @patch("src.data_prep.load.pl.scan_parquet")
    def test_download_parquet_to_cache_filename_extraction(self, mock_scan):
        """Test correct filename extraction from dataset path."""
        mock_lazy_frame = Mock(spec=LazyFrame)
        mock_scan.return_value = mock_lazy_frame
        mock_df = Mock()
        mock_lazy_frame.collect.return_value = mock_df

        # Test with complex path
        dataset_path = "hf://datasets/vibingshu/2024_formula1_championship_dataset/data/train-00000-of-00001.parquet"
        cache_dir = "test_cache"

        result = download_parquet_to_cache(dataset_path, cache_dir)

        assert result == "test_cache/train-00000-of-00001.parquet"
        mock_df.write_parquet.assert_called_once_with(
            Path("test_cache/train-00000-of-00001.parquet")
        )
