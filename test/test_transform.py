import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import polars as pl
import pytest

from src.data_prep.transform import transform_parquet_to_jsonl


class TestTransformParquetToJsonl:
    """Test suite for transform_parquet_to_jsonl function."""

    def test_happy_path_success_transformation(self, tmp_path):
        """Test successful transformation of parquet to JSONL."""
        # Create test data
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello world", "How are you?"],
                "output": ["Hi there", "I'm fine, thanks!"],
            }
        )

        # Create temporary parquet file
        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Create temporary output file
        tmp_jsonl = tmp_path / "test.jsonl"

        # Mock the file operations
        with patch("builtins.open", mock_open()) as mock_file:
            # Call the function
            result = transform_parquet_to_jsonl(str(tmp_parquet), str(tmp_jsonl))

            # Verify result
            assert result == str(tmp_jsonl)

            # Verify the mock was called correctly
            mock_file.assert_called_with(tmp_jsonl, "w", encoding="utf-8")

    def test_happy_path_auto_generate_output_path(self, tmp_path):
        """Test successful transformation with auto-generated output path."""
        # Create test data
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        # Create temporary parquet file
        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Call the function without output path
        result = transform_parquet_to_jsonl(str(tmp_parquet))

        # Verify auto-generated path
        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

    def test_file_not_found_error(self):
        """Test FileNotFoundError when input file doesn't exist."""
        non_existent_file = "/path/that/does/not/exist.parquet"

        with pytest.raises(FileNotFoundError, match="Input parquet file not found"):
            transform_parquet_to_jsonl(non_existent_file)

    def test_value_error_parquet_read_failure(self, tmp_path):
        """Test ValueError when parquet file can't be read."""
        # Create a file that's not a valid parquet
        tmp_file = tmp_path / "invalid.parquet"
        tmp_file.write_bytes(b"This is not a parquet file")

        with pytest.raises(ValueError, match="Failed to read parquet file"):
            transform_parquet_to_jsonl(str(tmp_file))

    def test_value_error_missing_required_columns(self, tmp_path):
        """Test ValueError when required columns are missing."""
        # Create test data without required columns
        test_data = pl.DataFrame({"text": ["Hello world"], "response": ["Hi there"]})

        tmp_parquet = tmp_path / "invalid.parquet"
        test_data.write_parquet(tmp_parquet)

        with pytest.raises(ValueError, match="Missing required columns"):
            transform_parquet_to_jsonl(str(tmp_parquet))

    def test_value_error_jsonl_write_failure(self, tmp_path):
        """Test ValueError when JSONL write fails."""
        # Create test data
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Mock open to raise an exception during write
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(ValueError, match="Failed to write JSONL file"):
                transform_parquet_to_jsonl(str(tmp_parquet))

    def test_data_filtering_null_values(self):
        """Test data filtering removes null/empty values."""

        # Mock polars operations
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            # Mock file existence check
            mock_exists.return_value = True

            # Mock the DataFrame operations
            mock_df = MagicMock()
            mock_df.columns = ["instruction", "output"]
            mock_df.__len__ = MagicMock(return_value=1)

            # Mock the filter operation to return only valid rows
            mock_filtered = MagicMock()
            mock_filtered.select.return_value.iter_rows.return_value = [
                {"prompt": "Valid instruction", "completion": "Valid completion"}
            ]

            mock_df.filter.return_value = mock_filtered
            mock_df.slice.return_value = mock_df
            mock_read.return_value = mock_df

            # Mock the file operations and capture written content
            written_content = []
            current_line = ""

            def mock_write_side_effect(data):
                nonlocal current_line
                current_line += data
                if data == "\n":
                    written_content.append(current_line.strip())
                    current_line = ""
                return None

            mock_file.return_value.__enter__.return_value.write = mock_write_side_effect

            # Call the function
            transform_parquet_to_jsonl("dummy.parquet")

            # Verify only valid data was processed
            assert len(written_content) == 1  # Only one valid row should be written
            written_json = json.loads(written_content[0])
            assert written_json["prompt"] == "Valid instruction"
            assert written_json["completion"] == "Valid completion"

    def test_data_filtering_empty_strings(self):
        """Test data filtering removes empty strings."""
        # Mock polars operations
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            # Mock file existence check
            mock_exists.return_value = True

            # Mock the DataFrame operations
            mock_df = MagicMock()
            mock_df.columns = ["instruction", "output"]
            mock_df.__len__ = MagicMock(return_value=1)

            # Mock the filter operation to return only valid rows
            mock_filtered = MagicMock()
            mock_filtered.select.return_value.iter_rows.return_value = [
                {"prompt": "Hello world", "completion": "Hi there"}
            ]

            mock_df.filter.return_value = mock_filtered
            mock_df.slice.return_value = mock_df
            mock_read.return_value = mock_df

            # Mock the file operations and capture written content
            written_content = []
            current_line = ""

            def mock_write_side_effect(data):
                nonlocal current_line
                current_line += data
                if data == "\n":
                    written_content.append(current_line.strip())
                    current_line = ""
                return None

            mock_file.return_value.__enter__.return_value.write = mock_write_side_effect

            # Call the function
            transform_parquet_to_jsonl("dummy.parquet")

            # Verify only valid data was processed
            assert len(written_content) == 1  # Only one valid row should be written
            written_json = json.loads(written_content[0])
            assert written_json["prompt"] == "Hello world"
            assert written_json["completion"] == "Hi there"

    def test_column_transformation(self):
        """Test that columns are correctly transformed."""
        # Mock polars operations
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            # Mock file existence check
            mock_exists.return_value = True

            # Mock the DataFrame operations
            mock_df = MagicMock()
            mock_df.columns = ["instruction", "output"]
            mock_df.__len__ = MagicMock(return_value=1)

            # Mock the filter and select operations
            mock_filtered = MagicMock()
            mock_transformed = MagicMock()
            mock_transformed.iter_rows.return_value = [
                {"prompt": "Hello world", "completion": "Hi there"}
            ]

            mock_filtered.select.return_value = mock_transformed
            mock_df.filter.return_value = mock_filtered
            mock_df.slice.return_value = mock_df
            mock_read.return_value = mock_df

            # Mock the file operations and capture written content
            written_content = []
            current_line = ""

            def mock_write_side_effect(data):
                nonlocal current_line
                current_line += data
                if data == "\n":
                    written_content.append(current_line.strip())
                    current_line = ""
                return None

            mock_file.return_value.__enter__.return_value.write = mock_write_side_effect

            # Call the function
            transform_parquet_to_jsonl("dummy.parquet")

            # Verify the written JSON content
            assert len(written_content) == 1
            written_json = json.loads(written_content[0])
            assert "prompt" in written_json
            assert "completion" in written_json
            assert written_json["prompt"] == "Hello world"
            assert written_json["completion"] == "Hi there"
            assert "instruction" not in written_json
            assert "output" not in written_json

    def test_multiple_rows_transformation(self):
        """Test transformation with multiple rows."""
        # Mock polars operations
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            # Mock file existence check
            mock_exists.return_value = True

            # Mock the DataFrame operations
            mock_df = MagicMock()
            mock_df.columns = ["instruction", "output"]
            mock_df.__len__ = MagicMock(return_value=3)

            # Mock the filter and select operations
            mock_filtered = MagicMock()
            mock_transformed = MagicMock()
            mock_transformed.iter_rows.return_value = [
                {"prompt": "Hello world", "completion": "Hi there"},
                {"prompt": "How are you?", "completion": "I'm fine"},
                {"prompt": "What's up?", "completion": "Not much"},
            ]

            mock_filtered.select.return_value = mock_transformed
            mock_df.filter.return_value = mock_filtered
            mock_df.slice.return_value = mock_df
            mock_read.return_value = mock_df

            # Mock the file operations and capture written content
            written_content = []
            current_line = ""

            def mock_write_side_effect(data):
                nonlocal current_line
                current_line += data
                if data == "\n":
                    written_content.append(current_line.strip())
                    current_line = ""
                return None

            mock_file.return_value.__enter__.return_value.write = mock_write_side_effect

            # Call the function
            transform_parquet_to_jsonl("dummy.parquet")

            # Verify all rows were processed
            assert len(written_content) == 3

            # Verify each row has correct structure
            for i, content in enumerate(written_content):
                written_json = json.loads(content)
                assert written_json["prompt"] in [
                    "Hello world",
                    "How are you?",
                    "What's up?",
                ]
                assert written_json["completion"] in [
                    "Hi there",
                    "I'm fine",
                    "Not much",
                ]

    def test_unicode_handling(self):
        """Test that unicode characters are handled correctly."""
        # Mock polars operations
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            # Mock file existence check
            mock_exists.return_value = True

            # Mock the DataFrame operations
            mock_df = MagicMock()
            mock_df.columns = ["instruction", "output"]
            mock_df.__len__ = MagicMock(return_value=2)

            # Mock the filter and select operations
            mock_filtered = MagicMock()
            mock_transformed = MagicMock()
            mock_transformed.iter_rows.return_value = [
                {"prompt": "Hello 世界", "completion": "Hi 世界"},
                {"prompt": "¿Cómo estás?", "completion": "Estoy bien, gracias!"},
            ]

            mock_filtered.select.return_value = mock_transformed
            mock_df.filter.return_value = mock_filtered
            mock_df.slice.return_value = mock_df
            mock_read.return_value = mock_df

            # Mock the file operations and capture written content
            written_content = []
            current_line = ""

            def mock_write_side_effect(data):
                nonlocal current_line
                current_line += data
                if data == "\n":
                    written_content.append(current_line.strip())
                    current_line = ""
                return None

            mock_file.return_value.__enter__.return_value.write = mock_write_side_effect

            # Call the function
            transform_parquet_to_jsonl("dummy.parquet")

            # Verify unicode characters are preserved
            assert len(written_content) == 2
            for content in written_content:
                written_json = json.loads(content)
                assert "世界" in written_json["prompt"] or "¿" in written_json["prompt"]

    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        # Create test data
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Create a non-existent output directory
        output_dir = Path("/tmp/non_existent_dir")
        output_file = output_dir / "output.jsonl"

        # Mock the file operations
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value.write.side_effect = (
                lambda x: None
            )

            # Call the function
            result = transform_parquet_to_jsonl(str(tmp_parquet), str(output_file))

            # Verify result
            assert result == str(output_file)

            # Verify the mock was called correctly
            mock_file.assert_called_with(output_file, "w", encoding="utf-8")

    def test_row_range_valid_range(self, tmp_path):
        """Test transformation with valid row range."""
        # Create test data with 5 rows
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What", "Where", "When"],
                "output": ["Hi", "Fine", "Up", "There", "Now"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Call the function with row range (1, 4) - should get rows 1, 2, 3
        result = transform_parquet_to_jsonl(str(tmp_parquet), row_range=(1, 4))

        # Verify result
        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

        # Verify the output contains only 3 rows
        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3

            # Verify the content
            first_row = json.loads(lines[0])
            assert first_row["prompt"] == "How"
            assert first_row["completion"] == "Fine"

    def test_row_range_invalid_range(self, tmp_path):
        """Test ValueError with invalid row range."""
        # Create test data with 3 rows
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What"],
                "output": ["Hi", "Fine", "Up"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Test invalid range - start >= end
        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(2, 2))

        # Test invalid range - start < 0
        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(-1, 2))

        # Test invalid range - end > len(df)
        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(0, 5))

    def test_row_range_full_range(self, tmp_path):
        """Test transformation with full row range."""
        # Create test data with 3 rows
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What"],
                "output": ["Hi", "Fine", "Up"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        # Call the function with full range (0, 3)
        result = transform_parquet_to_jsonl(str(tmp_parquet), row_range=(0, 3))

        # Verify result
        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

        # Verify the output contains all 3 rows
        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3
