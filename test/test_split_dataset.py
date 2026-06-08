import json
from pathlib import Path

from src.data_prep.transform import split_dataset


def _make_jsonl(path: Path, rows: list[dict]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")
    return str(path)


class TestSplitDataset:
    def _qa_row(self, i: int) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"Question {i}?"},
                {"role": "assistant", "content": f"Answer {i}."},
            ]
        }

    def test_creates_train_and_valid_files(self, tmp_path):
        rows = [self._qa_row(i) for i in range(50)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "output"

        result = split_dataset(jsonl_path, str(output_dir))

        assert len(result) == 2
        train_path, valid_path = result
        assert Path(train_path).exists()
        assert Path(valid_path).exists()
        assert Path(train_path).name == "train.jsonl"
        assert Path(valid_path).name == "valid.jsonl"

    def test_split_ratios_small_dataset(self, tmp_path):
        rows = [self._qa_row(i) for i in range(100)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "output"

        train_path, valid_path = split_dataset(jsonl_path, str(output_dir))

        with open(train_path) as f:
            train_count = sum(1 for line in f if line.strip())
        with open(valid_path) as f:
            valid_count = sum(1 for line in f if line.strip())

        assert train_count + valid_count == 100
        assert valid_count == 15

    def test_split_ratios_medium_dataset(self, tmp_path):
        rows = [self._qa_row(i) for i in range(500)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "output"

        train_path, valid_path = split_dataset(jsonl_path, str(output_dir))

        with open(valid_path) as f:
            valid_count = sum(1 for line in f if line.strip())

        assert valid_count == 50

    def test_split_ratios_large_dataset(self, tmp_path):
        rows = [self._qa_row(i) for i in range(2000)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "output"

        train_path, valid_path = split_dataset(jsonl_path, str(output_dir))

        with open(valid_path) as f:
            valid_count = sum(1 for line in f if line.strip())

        assert valid_count == 100

    def test_output_dir_created_if_missing(self, tmp_path):
        rows = [self._qa_row(i) for i in range(10)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "nested" / "dir"

        train_path, valid_path = split_dataset(jsonl_path, str(output_dir))

        assert Path(train_path).exists()
        assert Path(valid_path).exists()

    def test_returns_tuple_of_paths(self, tmp_path):
        rows = [self._qa_row(i) for i in range(10)]
        jsonl_path = _make_jsonl(tmp_path / "data.jsonl", rows)
        output_dir = tmp_path / "output"

        train_path, valid_path = split_dataset(jsonl_path, str(output_dir))

        assert isinstance(train_path, str)
        assert isinstance(valid_path, str)
