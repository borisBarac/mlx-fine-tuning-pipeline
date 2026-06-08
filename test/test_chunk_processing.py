import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from src.data_prep.transform import process_parquet_in_chunks


def _make_qa_parquet(path: Path, n: int) -> str:
    df = pl.DataFrame(
        {
            "instruction": [f"Question {i}?" for i in range(n)],
            "output": [f"Answer {i}." for i in range(n)],
        }
    )
    df.write_parquet(path)
    return str(path)


class TestProcessParquetInChunks:
    @patch("src.data_prep.transform.parallel_map")
    def test_splits_into_chunks(self, mock_parallel_map, tmp_path):
        parquet_path = _make_qa_parquet(tmp_path / "data.parquet", 250)
        temp_dir = tmp_path / "chunks"

        def fake_parallel_map(fn, inputs):
            results = []
            for inp in inputs:
                chunk_output = Path(temp_dir) / f"chunk_{inp[0]:04d}.jsonl"
                chunk_output.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_output, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "Q"},
                                    {"role": "assistant", "content": "A"},
                                ]
                            }
                        )
                        + "\n"
                    )
                results.append(
                    {
                        "chunk_idx": inp[0],
                        "start_row": inp[1][0],
                        "end_row": inp[1][1],
                        "output_path": str(chunk_output),
                        "processing_time": 0.1,
                        "success": True,
                        "error": None,
                    }
                )
            return results

        mock_parallel_map.side_effect = fake_parallel_map

        result = process_parquet_in_chunks(
            parquet_path, chunk_size=100, temp_dir=str(temp_dir)
        )

        assert len(result["chunk_files"]) > 0
        assert len(result["failed_chunks"]) == 0
        assert result["num_chunks"] == 3

    @patch("src.data_prep.transform.parallel_map")
    def test_raises_on_all_failures(self, mock_parallel_map, tmp_path):
        parquet_path = _make_qa_parquet(tmp_path / "data.parquet", 10)
        temp_dir = tmp_path / "chunks"

        mock_parallel_map.return_value = [
            {
                "chunk_idx": 0,
                "start_row": 0,
                "end_row": 10,
                "output_path": None,
                "processing_time": 0.1,
                "success": False,
                "error": "Something failed",
            }
        ]

        with pytest.raises(Exception, match="chunks failed processing"):
            process_parquet_in_chunks(
                parquet_path, chunk_size=100, temp_dir=str(temp_dir)
            )

    @patch("src.data_prep.transform.parallel_map")
    def test_returns_chunk_file_paths(self, mock_parallel_map, tmp_path):
        parquet_path = _make_qa_parquet(tmp_path / "data.parquet", 50)
        temp_dir = tmp_path / "chunks"

        def fake_parallel_map(fn, inputs):
            results = []
            for inp in inputs:
                chunk_output = Path(temp_dir) / f"chunk_{inp[0]:04d}.jsonl"
                chunk_output.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_output, "w") as f:
                    f.write(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "Q"},
                                    {"role": "assistant", "content": "A"},
                                ]
                            }
                        )
                        + "\n"
                    )
                results.append(
                    {
                        "chunk_idx": inp[0],
                        "start_row": inp[1][0],
                        "end_row": inp[1][1],
                        "output_path": str(chunk_output),
                        "processing_time": 0.1,
                        "success": True,
                        "error": None,
                    }
                )
            return results

        mock_parallel_map.side_effect = fake_parallel_map

        result = process_parquet_in_chunks(
            parquet_path, chunk_size=100, temp_dir=str(temp_dir)
        )

        for f in result["chunk_files"]:
            assert Path(f).exists()
