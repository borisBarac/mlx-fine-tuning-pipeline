import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def _make_flow_with_converted_parquet(tmp_path):
    from data_prep_flow import DataPrepFlow

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sample.txt").write_text(
        "Machine learning is a subset of artificial intelligence "
        "that enables systems to learn and improve from experience without being "
        "explicitly programmed. It focuses on developing algorithms.\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "converted"

    flow = MagicMock(spec=DataPrepFlow)
    flow.docs_path_str = str(docs_dir)
    flow.docs_output_str = str(output_dir)
    flow.teacher_model = "test-model"
    flow.num_examples = 2
    flow.generation_chunk_size = 2000
    flow.api_base = "https://api.example.com/v1"
    flow.api_key = "test-key"
    return flow, output_dir


class TestPipelineGenerateQaStep:
    @patch(
        "data_prep_flow.pl.read_parquet", return_value=pl.DataFrame({"a": range(42)})
    )
    @patch("data_prep_flow.generate_dataset")
    @patch("openai.OpenAI")
    def test_sets_parquet_path_str(
        self, mock_openai_cls, mock_generate, mock_read_parquet, tmp_path
    ):
        from data_prep_flow import DataPrepFlow

        flow, output_dir = _make_flow_with_converted_parquet(tmp_path)

        DataPrepFlow.convert_documents(flow)
        assert hasattr(flow, "converted_parquet_path")

        generated_path = str(tmp_path / "generated" / "generated_qa.parquet")
        mock_generate.return_value = generated_path

        DataPrepFlow.generate_qa(flow)

        assert flow.parquet_path_str == generated_path
        assert flow.total_rows == 42
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs["model"] == "test-model"
        assert call_kwargs.kwargs["num_examples"] == 2
        flow.next.assert_called_with(flow.process_chunks)

    @patch(
        "data_prep_flow.pl.read_parquet", return_value=pl.DataFrame({"a": range(10)})
    )
    @patch("data_prep_flow.generate_dataset")
    @patch("openai.OpenAI")
    def test_creates_openai_client_with_params(
        self, mock_openai_cls, mock_generate, mock_read_parquet, tmp_path
    ):
        from data_prep_flow import DataPrepFlow

        flow, output_dir = _make_flow_with_converted_parquet(tmp_path)
        DataPrepFlow.convert_documents(flow)

        mock_generate.return_value = "/tmp/out.parquet"

        DataPrepFlow.generate_qa(flow)

        mock_openai_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )

    @patch(
        "data_prep_flow.pl.read_parquet", return_value=pl.DataFrame({"a": range(10)})
    )
    @patch("data_prep_flow.generate_dataset")
    @patch("openai.OpenAI")
    def test_uses_env_var_when_api_key_empty(
        self, mock_openai_cls, mock_generate, mock_read_parquet, tmp_path
    ):
        from data_prep_flow import DataPrepFlow

        flow, output_dir = _make_flow_with_converted_parquet(tmp_path)
        flow.api_key = ""
        DataPrepFlow.convert_documents(flow)

        mock_generate.return_value = "/tmp/out.parquet"

        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            DataPrepFlow.generate_qa(flow)

        mock_openai_cls.assert_called_once_with(
            api_key="env-key",
            base_url="https://api.example.com/v1",
        )

    @patch("openai.OpenAI")
    def test_raises_when_no_api_key(self, mock_openai_cls, tmp_path):
        from data_prep_flow import DataPrepFlow

        flow, output_dir = _make_flow_with_converted_parquet(tmp_path)
        flow.api_key = ""
        DataPrepFlow.convert_documents(flow)

        with patch.dict("os.environ", {}, clear=False):
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            with pytest.raises(ValueError, match="api-key"):
                DataPrepFlow.generate_qa(flow)

    @patch(
        "data_prep_flow.pl.read_parquet", return_value=pl.DataFrame({"a": range(10)})
    )
    @patch("data_prep_flow.generate_dataset")
    @patch("openai.OpenAI")
    def test_output_dir_is_inside_converted(
        self, mock_openai_cls, mock_generate, mock_read_parquet, tmp_path
    ):
        from data_prep_flow import DataPrepFlow

        flow, output_dir = _make_flow_with_converted_parquet(tmp_path)
        DataPrepFlow.convert_documents(flow)

        mock_generate.return_value = "/tmp/out.parquet"

        DataPrepFlow.generate_qa(flow)

        call_kwargs = mock_generate.call_args
        assert "generated" in call_kwargs.kwargs["output_dir"]
