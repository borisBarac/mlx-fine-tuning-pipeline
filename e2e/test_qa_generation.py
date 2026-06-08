import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from data_prep.convert import convert_documents
from data_prep.generate import generate_dataset
from data_prep.transform import split_dataset, transform_parquet_to_jsonl

E2E_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = E2E_DIR.parent
OUT_DIR = E2E_DIR / "out"
FIXTURE_PDF = E2E_DIR / "LordRings.pdf"


@pytest.fixture
def output_dirs(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shutil.copy2(FIXTURE_PDF, docs_dir / "LordRings.pdf")

    converted_dir = OUT_DIR / "converted"
    generated_dir = OUT_DIR / "generated"
    for d in [converted_dir, generated_dir, OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    return docs_dir, converted_dir, generated_dir


def test_qa_generation_pipeline_e2e(output_dirs, openrouter_client, teacher_model):
    docs_dir, converted_dir, generated_dir = output_dirs

    converted_parquet = convert_documents(
        str(docs_dir), str(converted_dir), num_threads=1
    )

    assert Path(converted_parquet).exists()
    converted_df = pl.read_parquet(converted_parquet)
    assert set(converted_df.columns) == {
        "source_file",
        "page_number",
        "section_type",
        "text_content",
    }
    assert len(converted_df) > 0

    generated_parquet = generate_dataset(
        input_parquet=converted_parquet,
        output_dir=str(generated_dir),
        client=openrouter_client,
        model=teacher_model,
        num_examples=20,
        chunk_size=2000,
    )

    assert Path(generated_parquet).exists()
    qa_df = pl.read_parquet(generated_parquet)
    assert set(qa_df.columns) == {"instruction", "output"}
    assert len(qa_df) > 0
    for row in qa_df.iter_rows(named=True):
        assert row["instruction"].strip() != ""
        assert row["output"].strip() != ""

    jsonl_path = str(OUT_DIR / "all.jsonl")
    transform_parquet_to_jsonl(generated_parquet, jsonl_path)

    assert Path(jsonl_path).exists()
    with open(jsonl_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) > 0
    for line in lines:
        record = json.loads(line)
        assert "messages" in record
        assert len(record["messages"]) == 2
        assert record["messages"][0]["role"] == "user"
        assert record["messages"][1]["role"] == "assistant"
        assert record["messages"][0]["content"].strip() != ""
        assert record["messages"][1]["content"].strip() != ""

    train_path, valid_path = split_dataset(
        jsonl_path, str(OUT_DIR)
    )

    assert Path(train_path).exists()
    assert Path(valid_path).exists()

    with open(train_path) as f:
        train_count = sum(1 for line in f if line.strip())
    with open(valid_path) as f:
        valid_count = sum(1 for line in f if line.strip())

    assert train_count + valid_count == len(qa_df)
    assert train_count > 0
    assert valid_count >= 0
