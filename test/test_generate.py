import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data_prep.generate import (
    filter_qa_pairs,
    generate_dataset,
    generate_qa_pairs,
    _build_chunks,
)


def _make_converted_parquet(tmp_path: Path, rows: list[dict] | None = None) -> str:
    if rows is None:
        rows = [
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn and improve from experience without being "
                "explicitly programmed. It focuses on developing algorithms that can access "
                "data and use it to learn for themselves.",
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": "Deep learning is a subset of machine learning that uses "
                "neural networks with many layers to analyze various factors of data. "
                "Deep learning algorithms use large amounts of data and complex neural "
                "network architectures to deliver accurate results.",
            },
        ]
    df = pl.DataFrame(
        rows,
        schema={
            "source_file": pl.String,
            "page_number": pl.Int64,
            "section_type": pl.String,
            "text_content": pl.String,
        },
    )
    path = tmp_path / "converted.parquet"
    df.write_parquet(path)
    return str(path)


def _mock_client(response_qa_pairs: list[dict]) -> MagicMock:
    client = MagicMock()
    message = MagicMock()
    message.content = json.dumps({"qa_pairs": response_qa_pairs})
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


class TestGenerateQaPairs:
    def test_returns_list_of_qa_dicts(self):
        client = _mock_client(
            [
                {
                    "question": "What is machine learning?",
                    "answer": "Machine learning is a subset of AI.",
                },
            ]
        )
        result = generate_qa_pairs("some content", client, "test-model", 1)
        assert len(result) == 1
        assert result[0]["question"] == "What is machine learning?"
        assert result[0]["answer"] == "Machine learning is a subset of AI."

    def test_sends_correct_model_and_format(self):
        client = _mock_client([])
        generate_qa_pairs("content", client, "deepseek-chat", 3)
        call_args = client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "deepseek-chat"
        assert call_args.kwargs["response_format"] == {"type": "json_object"}

    def test_sends_system_and_user_messages(self):
        client = _mock_client([])
        generate_qa_pairs("my content here", client, "model", 5)
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "my content here" in messages[1]["content"]
        assert "5" in messages[1]["content"]

    def test_handles_empty_response(self):
        client = _mock_client([])
        result = generate_qa_pairs("content", client, "model", 3)
        assert result == []

    def test_skips_pairs_missing_fields(self):
        client = _mock_client(
            [
                {"question": "Valid?", "answer": "Valid answer here."},
                {"question": "No answer"},
                {"answer": "No question"},
            ]
        )
        result = generate_qa_pairs("content", client, "model", 3)
        assert len(result) == 1
        assert result[0]["question"] == "Valid?"

    def test_handles_malformed_json(self):
        client = MagicMock()
        message = MagicMock()
        message.content = "not valid json"
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        with pytest.raises(json.JSONDecodeError):
            generate_qa_pairs("content", client, "model", 1)

    def test_handles_null_content(self):
        client = MagicMock()
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        response = MagicMock()
        response.choices = [choice]
        client.chat.completions.create.return_value = response

        result = generate_qa_pairs("content", client, "model", 1)
        assert result == []


class TestFilterQaPairs:
    def test_keeps_valid_pairs(self):
        pairs = [
            {
                "question": "What is the primary benefit of machine learning in production?",
                "answer": "The primary benefit is automated decision making at scale, enabling organizations to process large volumes of data efficiently and accurately.",
            },
        ]
        result = filter_qa_pairs(pairs)
        assert len(result) == 1

    def test_removes_short_questions(self):
        pairs = [
            {
                "question": "What?",
                "answer": "This is a long enough answer with many words in it.",
            },
        ]
        result = filter_qa_pairs(pairs)
        assert len(result) == 0

    def test_removes_short_answers(self):
        pairs = [
            {
                "question": "What is the main advantage of this approach?",
                "answer": "Yes.",
            },
        ]
        result = filter_qa_pairs(pairs)
        assert len(result) == 0

    def test_removes_pairs_missing_fields(self):
        pairs = [
            {"question": "What is the main advantage of this approach?"},
            {"answer": "This is a detailed answer with many words in it."},
        ]
        result = filter_qa_pairs(pairs)
        assert len(result) == 0

    def test_custom_thresholds(self):
        pairs = [
            {"question": "Short q?", "answer": "Short answer."},
        ]
        result = filter_qa_pairs(pairs, min_question_words=2, min_answer_words=2)
        assert len(result) == 1

    def test_mixed_valid_and_invalid(self):
        pairs = [
            {
                "question": "What is the primary benefit of this system?",
                "answer": "The benefit is improved efficiency across all operations, reducing manual overhead and increasing overall productivity significantly for the entire organization.",
            },
            {"question": "Huh?", "answer": "Short."},
            {
                "question": "How does this process work in practice?",
                "answer": "It works by applying a series of transformations to the input data, processing each element sequentially through the pipeline stages.",
            },
        ]
        result = filter_qa_pairs(pairs)
        assert len(result) == 2


class TestBuildChunks:
    def test_splits_into_chunks_by_word_count(self, tmp_path):
        rows = [
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(50)),
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(50, 100)),
            },
        ]
        parquet_path = _make_converted_parquet(tmp_path, rows)
        df = pl.read_parquet(parquet_path)
        chunks = _build_chunks(df, chunk_size=60)
        assert len(chunks) >= 2

    def test_includes_source_metadata(self, tmp_path):
        rows = [
            {
                "source_file": "mydoc.pdf",
                "page_number": 3,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(60)),
            },
        ]
        parquet_path = _make_converted_parquet(tmp_path, rows)
        df = pl.read_parquet(parquet_path)
        chunks = _build_chunks(df, chunk_size=2000)
        assert len(chunks) == 1
        assert "mydoc.pdf" in chunks[0]["text"]
        assert "3" in chunks[0]["text"]

    def test_splits_on_section_header(self, tmp_path):
        rows = [
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(60)),
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "section_header",
                "text_content": "New Section",
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(60, 120)),
            },
        ]
        parquet_path = _make_converted_parquet(tmp_path, rows)
        df = pl.read_parquet(parquet_path)
        chunks = _build_chunks(df, chunk_size=2000)
        assert len(chunks) == 2


class TestGenerateDataset:
    def test_outputs_parquet_with_instruction_and_output(self, tmp_path):
        parquet_path = _make_converted_parquet(tmp_path)
        client = _mock_client(
            [
                {
                    "question": "What is machine learning in the context of modern technology?",
                    "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from experience without being explicitly programmed automatically.",
                },
                {
                    "question": "What is deep learning and how does it relate to machine learning?",
                    "answer": "Deep learning is a subset of machine learning that uses neural networks with many layers to analyze various factors of data effectively.",
                },
            ]
        )

        with patch("src.data_prep.generate.time.sleep"):
            output = generate_dataset(
                parquet_path,
                str(tmp_path / "out"),
                client,
                "test-model",
                2,
                chunk_size=2000,
            )

        assert Path(output).exists()
        df = pl.read_parquet(output)
        assert set(df.columns) == {"instruction", "output"}
        assert len(df) > 0
        assert all(row["instruction"] != "" for row in df.iter_rows(named=True))
        assert all(row["output"] != "" for row in df.iter_rows(named=True))

    def test_distributes_examples_proportionally(self, tmp_path):
        rows = [
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(100)),
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "section_header",
                "text_content": "Section Two",
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(50)),
            },
        ]
        parquet_path = _make_converted_parquet(tmp_path, rows)
        client = _mock_client(
            [
                {
                    "question": "What is this content about in general terms?",
                    "answer": "This is a detailed answer about the content provided in the document section which covers multiple important topics very comprehensively.",
                },
            ]
        )

        with patch("src.data_prep.generate.time.sleep"):
            generate_dataset(
                parquet_path,
                str(tmp_path / "out"),
                client,
                "test-model",
                4,
                chunk_size=2000,
            )

        assert client.chat.completions.create.call_count >= 2

    def test_raises_on_zero_examples(self, tmp_path):
        parquet_path = _make_converted_parquet(tmp_path)
        client = _mock_client([])

        with patch("src.data_prep.generate.time.sleep"):
            with pytest.raises(RuntimeError, match="Zero QA pairs"):
                generate_dataset(
                    parquet_path,
                    str(tmp_path / "out"),
                    client,
                    "test-model",
                    10,
                    chunk_size=2000,
                )


class TestGenerateDatasetErrorHandling:
    def test_continues_on_api_error(self, tmp_path):
        rows = [
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(60)),
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "section_header",
                "text_content": "Deep Learning",
            },
            {
                "source_file": "doc.txt",
                "page_number": 1,
                "section_type": "text",
                "text_content": " ".join(f"word{i}" for i in range(60, 120)),
            },
        ]
        parquet_path = _make_converted_parquet(tmp_path, rows)
        client = MagicMock()

        good_message = MagicMock()
        good_message.content = json.dumps(
            {
                "qa_pairs": [
                    {
                        "question": "What is machine learning in simple terms for beginners?",
                        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from experience without explicit programming entirely automatically.",
                    },
                ]
            }
        )
        good_choice = MagicMock()
        good_choice.message = good_message
        good_response = MagicMock()
        good_response.choices = [good_choice]

        client.chat.completions.create.side_effect = [
            Exception("API timeout"),
            good_response,
        ]

        with patch("src.data_prep.generate.time.sleep"):
            output = generate_dataset(
                parquet_path,
                str(tmp_path / "out"),
                client,
                "test-model",
                2,
                chunk_size=2000,
            )

        df = pl.read_parquet(output)
        assert len(df) >= 1

    def test_raises_when_fewer_than_half_requested(self, tmp_path):
        parquet_path = _make_converted_parquet(tmp_path)
        client = _mock_client(
            [
                {
                    "question": "What is this content about in the given document?",
                    "answer": "A short answer that is long enough to pass the default minimum word count filter threshold for quality assurance testing purposes.",
                },
            ]
        )

        with patch("src.data_prep.generate.time.sleep"):
            with pytest.raises(RuntimeError, match="QA pairs generated"):
                generate_dataset(
                    parquet_path,
                    str(tmp_path / "out"),
                    client,
                    "test-model",
                    100,
                    chunk_size=2000,
                )

    def test_handles_rate_limit_on_all_chunks(self, tmp_path):
        parquet_path = _make_converted_parquet(tmp_path)
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        with patch("src.data_prep.generate.time.sleep"):
            with pytest.raises(RuntimeError, match="Zero QA pairs"):
                generate_dataset(
                    parquet_path,
                    str(tmp_path / "out"),
                    client,
                    "test-model",
                    5,
                    chunk_size=2000,
                )

    def test_empty_parquet_raises(self, tmp_path):
        df = pl.DataFrame(
            {
                "source_file": [],
                "page_number": [],
                "section_type": [],
                "text_content": [],
            },
            schema={
                "source_file": pl.String,
                "page_number": pl.Int64,
                "section_type": pl.String,
                "text_content": pl.String,
            },
        )
        path = tmp_path / "empty.parquet"
        df.write_parquet(path)

        client = _mock_client([])
        with pytest.raises(RuntimeError, match="No text chunks"):
            generate_dataset(
                str(path),
                str(tmp_path / "out"),
                client,
                "test-model",
                5,
                chunk_size=2000,
            )
