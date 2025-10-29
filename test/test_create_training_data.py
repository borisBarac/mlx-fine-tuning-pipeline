import json

from src.data_prep.create_training_data import merge_jsonl_files


class TestMergeJsonlFiles:
    """Test suite for merge_jsonl_files function."""

    def test_merge_valid_files_success(self, tmp_path):
        """Test successful merge of valid JSONL files."""
        # Create temporary valid JSONL files
        # Create first valid JSONL file
        file1_path = tmp_path / "file1.jsonl"
        with open(file1_path, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello", "completion": "Hi"}\n')
            f.write('{"prompt": "How are you?", "completion": "Fine"}\n')

        # Create second valid JSONL file
        file2_path = tmp_path / "file2.jsonl"
        with open(file2_path, "w", encoding="utf-8") as f:
            f.write('{"prompt": "What\'s up?", "completion": "Not much"}\n')

        # Create output file path
        output_path = tmp_path / "merged.jsonl"

        # Call the function
        result = merge_jsonl_files([str(file1_path), str(file2_path)], str(output_path))

        # Verify result
        assert result["output_file"] == str(output_path)
        assert result["files_merged"] == 2
        assert result["files_skipped"] == 0
        assert result["total_lines_merged"] == 3
        assert len(result["valid_files"]) == 2
        assert len(result["invalid_files"]) == 0

        # Verify output file content
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3

            # Verify each line is valid JSON
            for line in lines:
                json.loads(line.strip())

    def test_merge_with_invalid_files(self, tmp_path):
        """Test merge with some invalid JSONL files."""
        # Create valid JSONL file
        valid_file = tmp_path / "valid.jsonl"
        with open(valid_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello", "completion": "Hi"}\n')

        # Create invalid JSONL file (malformed JSON)
        invalid_file = tmp_path / "invalid.jsonl"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello", "completion": "Hi"\n')  # Missing closing brace

        # Create non-existent file
        non_existent_file = tmp_path / "non_existent.jsonl"

        # Create output file path
        output_path = tmp_path / "merged.jsonl"

        # Call the function
        result = merge_jsonl_files(
            [str(valid_file), str(invalid_file), str(non_existent_file)],
            str(output_path),
        )

        # Verify result
        assert result["files_merged"] == 1
        assert result["files_skipped"] == 2
        assert result["total_lines_merged"] == 1
        assert len(result["valid_files"]) == 1
        assert len(result["invalid_files"]) == 2
        assert str(invalid_file) in result["invalid_files"]
        assert str(non_existent_file) in result["invalid_files"]

        # Verify output file contains only valid content
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            json.loads(lines[0].strip())

    def test_merge_empty_files_list(self, tmp_path):
        """Test merge with empty files list."""
        output_path = tmp_path / "merged.jsonl"

        result = merge_jsonl_files([], str(output_path))

        assert result["files_merged"] == 0
        assert result["files_skipped"] == 0
        assert result["total_lines_merged"] == 0
        assert len(result["valid_files"]) == 0
        assert len(result["invalid_files"]) == 0

    def test_merge_all_invalid_files(self, tmp_path):
        """Test merge when all files are invalid."""
        # Create invalid JSONL file
        invalid_file = tmp_path / "invalid.jsonl"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write("This is not JSON at all\n")

        # Create non-existent file
        non_existent_file = tmp_path / "non_existent.jsonl"

        output_path = tmp_path / "merged.jsonl"

        result = merge_jsonl_files(
            [str(invalid_file), str(non_existent_file)], str(output_path)
        )

        assert result["files_merged"] == 0
        assert result["files_skipped"] == 2
        assert result["total_lines_merged"] == 0
        assert len(result["valid_files"]) == 0
        assert len(result["invalid_files"]) == 2

    def test_merge_empty_jsonl_files(self, tmp_path):
        """Test merge with empty JSONL files."""
        # Create empty JSONL file
        empty_file = tmp_path / "empty.jsonl"
        empty_file.touch()

        # Create JSONL file with only whitespace
        whitespace_file = tmp_path / "whitespace.jsonl"
        with open(whitespace_file, "w", encoding="utf-8") as f:
            f.write("   \n\n\t\n")

        output_path = tmp_path / "merged.jsonl"

        result = merge_jsonl_files(
            [str(empty_file), str(whitespace_file)], str(output_path)
        )

        # Empty files should be considered invalid (no valid JSON lines)
        assert result["files_merged"] == 0
        assert result["files_skipped"] == 2

    def test_merge_unicode_handling(self, tmp_path):
        """Test merge with unicode characters in JSONL files."""
        # Create JSONL file with unicode content
        unicode_file = tmp_path / "unicode.jsonl"
        with open(unicode_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello 世界", "completion": "Hi 世界"}\n')
            f.write('{"prompt": "¿Cómo estás?", "completion": "Estoy bien"}\n')

        output_path = tmp_path / "merged.jsonl"

        result = merge_jsonl_files([str(unicode_file)], str(output_path))

        assert result["files_merged"] == 1
        assert result["total_lines_merged"] == 2

        # Verify unicode is preserved
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2

            first_line = json.loads(lines[0].strip())
            assert "世界" in first_line["prompt"]

            second_line = json.loads(lines[1].strip())
            assert "¿" in second_line["prompt"]

    def test_merge_large_file_validation(self, tmp_path):
        """Test that validation only checks first 5 lines of large files."""
        # Create JSONL file with many lines
        large_file = tmp_path / "large.jsonl"
        with open(large_file, "w", encoding="utf-8") as f:
            for i in range(100):
                f.write(f'{{"prompt": "Line {i}", "completion": "Response {i}"}}\n')

        output_path = tmp_path / "merged.jsonl"

        result = merge_jsonl_files([str(large_file)], str(output_path))

        assert result["files_merged"] == 1
        assert result["total_lines_merged"] == 100

    def test_merge_file_path_types(self, tmp_path):
        """Test merge with different path types (str and Path objects)."""
        # Create valid JSONL file
        valid_file = tmp_path / "valid.jsonl"
        with open(valid_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello", "completion": "Hi"}\n')

        output_path = tmp_path / "merged.jsonl"

        # Test with mix of str and Path objects
        result = merge_jsonl_files(
            [str(valid_file), valid_file],  # Mix of str and Path
            str(output_path),
        )

        assert result["files_merged"] == 2
        assert result["total_lines_merged"] == 2

    def test_merge_output_file_creation(self, tmp_path):
        """Test that output file is created and can be written."""
        # Create valid JSONL file
        valid_file = tmp_path / "valid.jsonl"
        with open(valid_file, "w", encoding="utf-8") as f:
            f.write('{"prompt": "Hello", "completion": "Hi"}\n')

        # Use nested directory for output
        output_dir = tmp_path / "output" / "nested"
        output_path = output_dir / "merged.jsonl"

        result = merge_jsonl_files([str(valid_file)], str(output_path))

        assert result["files_merged"] == 1
        assert output_path.exists()

        # Verify content
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            json.loads(lines[0].strip())
