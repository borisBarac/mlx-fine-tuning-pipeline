import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class _FlowProxy:
    def __init__(self, **kwargs):
        self.docs_path = kwargs.get("docs_path", "")
        self.docs_output = kwargs.get("docs_output", "")
        self.data_output = kwargs.get("data_output", "")
        self.script_dir = kwargs.get("script_dir", Path(__file__).parent.parent / "src")
        self.chunk_size = kwargs.get("chunk_size", 100)
        self.next = MagicMock()
        self.convert_documents = MagicMock()

    def _validate_docs_path(self):
        from data_prep_flow import DataPrepFlow

        DataPrepFlow._validate_docs_path(self)


class TestPipelineStartValidatesDocsPath:
    def test_raises_when_docs_path_empty(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        flow = _FlowProxy(docs_path="")
        with pytest.raises(ValueError, match="--docs-path is required"):
            DataPrepFlow._validate_docs_path(flow)

    def test_raises_when_docs_path_not_directory(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        flow = _FlowProxy(docs_path=str(tmp_path / "nonexistent"))
        with pytest.raises(ValueError, match="must be an existing directory"):
            DataPrepFlow._validate_docs_path(flow)

    def test_passes_with_valid_docs_path(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")

        flow = _FlowProxy(docs_path=str(docs_dir))
        DataPrepFlow.start(flow)
        assert flow.docs_path_str == str(docs_dir)
        flow.next.assert_called_once()

    def test_defaults_docs_output_to_converted_subdir(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")

        flow = _FlowProxy(docs_path=str(docs_dir), docs_output="")
        DataPrepFlow.start(flow)
        assert flow.docs_output_str == str(docs_dir / "converted")

    def test_uses_custom_docs_output(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")
        custom_output = tmp_path / "custom_output"

        flow = _FlowProxy(docs_path=str(docs_dir), docs_output=str(custom_output))
        DataPrepFlow.start(flow)
        assert flow.docs_output_str == str(custom_output)


class TestPipelineConvertDocumentsStep:
    def test_sets_converted_parquet_path(self, tmp_path):
        from data_prep_flow import DataPrepFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("Hello world\n")

        output_dir = tmp_path / "output"

        flow = MagicMock()
        flow.docs_path_str = str(docs_dir)
        flow.docs_output_str = str(output_dir)

        DataPrepFlow.convert_documents(flow)

        assert hasattr(flow, "converted_parquet_path")
        assert flow.converted_parquet_path.endswith("converted.parquet")
        assert Path(flow.converted_parquet_path).exists()
        flow.next.assert_called_once()
