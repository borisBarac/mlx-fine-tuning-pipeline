import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import polars as pl
import pytest

from src.data_prep.transform import transform_parquet_to_jsonl


def _make_mock_df(columns, rows, length):
    mock_df = MagicMock()
    mock_df.columns = columns
    mock_df.__len__ = MagicMock(return_value=length)
    mock_filtered = MagicMock()
    mock_filtered.select.return_value.iter_rows.return_value = rows
    mock_df.filter.return_value = mock_filtered
    mock_df.slice.return_value = mock_df
    return mock_df


def _capture_writes(mock_file):
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
    return written_content


class TestTransformParquetToJsonl:
    """Test suite for transform_parquet_to_jsonl function."""

    def test_happy_path_success_transformation(self, tmp_path):
        """Test successful transformation of parquet to JSONL."""
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello world", "How are you?"],
                "output": ["Hi there", "I'm fine, thanks!"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        tmp_jsonl = tmp_path / "test.jsonl"

        with patch("builtins.open", mock_open()) as mock_file:
            result = transform_parquet_to_jsonl(str(tmp_parquet), str(tmp_jsonl))

            assert result == str(tmp_jsonl)
            mock_file.assert_called_with(tmp_jsonl, "w", encoding="utf-8")

    def test_happy_path_auto_generate_output_path(self, tmp_path):
        """Test successful transformation with auto-generated output path."""
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        result = transform_parquet_to_jsonl(str(tmp_parquet))

        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

    def test_file_not_found_error(self):
        """Test FileNotFoundError when input file doesn't exist."""
        non_existent_file = "/path/that/does/not/exist.parquet"

        with pytest.raises(FileNotFoundError, match="Input parquet file not found"):
            transform_parquet_to_jsonl(non_existent_file)

    def test_value_error_parquet_read_failure(self, tmp_path):
        """Test ValueError when parquet file can't be read."""
        tmp_file = tmp_path / "invalid.parquet"
        tmp_file.write_bytes(b"This is not a parquet file")

        with pytest.raises(ValueError, match="Failed to read parquet file"):
            transform_parquet_to_jsonl(str(tmp_file))

    def test_value_error_missing_required_columns(self, tmp_path):
        """Test ValueError when required columns are missing."""
        test_data = pl.DataFrame({"text": ["Hello world"], "response": ["Hi there"]})

        tmp_parquet = tmp_path / "invalid.parquet"
        test_data.write_parquet(tmp_parquet)

        with pytest.raises(ValueError, match="Missing required columns"):
            transform_parquet_to_jsonl(str(tmp_parquet))

    def test_value_error_jsonl_write_failure(self, tmp_path):
        """Test ValueError when JSONL write fails."""
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(ValueError, match="Failed to write JSONL file"):
                transform_parquet_to_jsonl(str(tmp_parquet))

    def test_data_filtering_null_values(self):
        """Test data filtering removes null/empty values."""
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_df = _make_mock_df(
                ["instruction", "output"],
                [{"instruction": "Valid instruction", "output": "Valid completion"}],
                1,
            )
            mock_read.return_value = mock_df
            written_content = _capture_writes(mock_file)

            transform_parquet_to_jsonl("dummy.parquet")

            assert len(written_content) == 1
            written_json = json.loads(written_content[0])
            assert written_json["messages"][0]["role"] == "user"
            assert written_json["messages"][0]["content"] == "Valid instruction"
            assert written_json["messages"][1]["role"] == "assistant"
            assert written_json["messages"][1]["content"] == "Valid completion"

    def test_data_filtering_empty_strings(self):
        """Test data filtering removes empty strings."""
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_df = _make_mock_df(
                ["instruction", "output"],
                [{"instruction": "Hello world", "output": "Hi there"}],
                1,
            )
            mock_read.return_value = mock_df
            written_content = _capture_writes(mock_file)

            transform_parquet_to_jsonl("dummy.parquet")

            assert len(written_content) == 1
            written_json = json.loads(written_content[0])
            assert written_json["messages"][0]["content"] == "Hello world"
            assert written_json["messages"][1]["content"] == "Hi there"

    def test_column_transformation(self):
        """Test that columns are correctly transformed to messages format."""
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_df = _make_mock_df(
                ["instruction", "output"],
                [{"instruction": "Hello world", "output": "Hi there"}],
                1,
            )
            mock_read.return_value = mock_df
            written_content = _capture_writes(mock_file)

            transform_parquet_to_jsonl("dummy.parquet")

            assert len(written_content) == 1
            written_json = json.loads(written_content[0])
            assert "messages" in written_json
            assert len(written_json["messages"]) == 2
            assert written_json["messages"][0] == {
                "role": "user",
                "content": "Hello world",
            }
            assert written_json["messages"][1] == {
                "role": "assistant",
                "content": "Hi there",
            }
            assert "instruction" not in written_json
            assert "output" not in written_json
            assert "prompt" not in written_json
            assert "completion" not in written_json

    def test_multiple_rows_transformation(self):
        """Test transformation with multiple rows."""
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_df = _make_mock_df(
                ["instruction", "output"],
                [
                    {"instruction": "Hello world", "output": "Hi there"},
                    {"instruction": "How are you?", "output": "I'm fine"},
                    {"instruction": "What's up?", "output": "Not much"},
                ],
                3,
            )
            mock_read.return_value = mock_df
            written_content = _capture_writes(mock_file)

            transform_parquet_to_jsonl("dummy.parquet")

            assert len(written_content) == 3

            for content in written_content:
                written_json = json.loads(content)
                user_msgs = [
                    m["content"]
                    for m in written_json["messages"]
                    if m["role"] == "user"
                ]
                asst_msgs = [
                    m["content"]
                    for m in written_json["messages"]
                    if m["role"] == "assistant"
                ]
                assert user_msgs[0] in [
                    "Hello world",
                    "How are you?",
                    "What's up?",
                ]
                assert asst_msgs[0] in [
                    "Hi there",
                    "I'm fine",
                    "Not much",
                ]

    def test_unicode_handling(self):
        """Test that unicode characters are handled correctly."""
        with (
            patch("src.data_prep.transform.pl.read_parquet") as mock_read,
            patch("builtins.open", mock_open()) as mock_file,
            patch("src.data_prep.transform.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_df = _make_mock_df(
                ["instruction", "output"],
                [
                    {"instruction": "Hello 世界", "output": "Hi 世界"},
                    {"instruction": "¿Cómo estás?", "output": "Estoy bien, gracias!"},
                ],
                2,
            )
            mock_read.return_value = mock_df
            written_content = _capture_writes(mock_file)

            transform_parquet_to_jsonl("dummy.parquet")

            assert len(written_content) == 2
            for content in written_content:
                written_json = json.loads(content)
                user_content = written_json["messages"][0]["content"]
                assert "世界" in user_content or "¿" in user_content

    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        test_data = pl.DataFrame(
            {"instruction": ["Hello world"], "output": ["Hi there"]}
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        output_dir = Path("/tmp/non_existent_dir")
        output_file = output_dir / "output.jsonl"

        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.return_value.__enter__.return_value.write.side_effect = (
                lambda x: None
            )

            result = transform_parquet_to_jsonl(str(tmp_parquet), str(output_file))

            assert result == str(output_file)
            mock_file.assert_called_with(output_file, "w", encoding="utf-8")

    def test_row_range_valid_range(self, tmp_path):
        """Test transformation with valid row range."""
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What", "Where", "When"],
                "output": ["Hi", "Fine", "Up", "There", "Now"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        result = transform_parquet_to_jsonl(str(tmp_parquet), row_range=(1, 4))

        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3

            first_row = json.loads(lines[0])
            assert first_row["messages"][0]["content"] == "How"
            assert first_row["messages"][1]["content"] == "Fine"

    def test_row_range_invalid_range(self, tmp_path):
        """Test ValueError with invalid row range."""
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What"],
                "output": ["Hi", "Fine", "Up"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(2, 2))

        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(-1, 2))

        with pytest.raises(ValueError, match="Invalid row range"):
            transform_parquet_to_jsonl(str(tmp_parquet), row_range=(0, 5))

    def test_row_range_full_range(self, tmp_path):
        """Test transformation with full row range."""
        test_data = pl.DataFrame(
            {
                "instruction": ["Hello", "How", "What"],
                "output": ["Hi", "Fine", "Up"],
            }
        )

        tmp_parquet = tmp_path / "test.parquet"
        test_data.write_parquet(tmp_parquet)

        result = transform_parquet_to_jsonl(str(tmp_parquet), row_range=(0, 3))

        expected_output = str(tmp_parquet.with_suffix(".jsonl"))
        assert result == expected_output

        with open(result, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3
