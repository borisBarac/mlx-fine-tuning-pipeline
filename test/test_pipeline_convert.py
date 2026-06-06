import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


class TestPipelineStartValidatesDocsPath:
    def _make_flow(self, docs_path="", docs_output=""):
        from pipeline import ParallelDataFlow

        flow = MagicMock(spec=ParallelDataFlow)
        flow.docs_path = docs_path
        flow.docs_output = docs_output
        flow.script_dir = Path(__file__).parent.parent / "src"
        flow.chunk_size = 100
        return flow

    def test_raises_when_docs_path_empty(self, tmp_path):
        from pipeline import ParallelDataFlow

        flow = self._make_flow(docs_path="")
        with pytest.raises(ValueError, match="--docs-path is required"):
            ParallelDataFlow.start(flow)

    def test_raises_when_docs_path_not_directory(self, tmp_path):
        from pipeline import ParallelDataFlow

        flow = self._make_flow(docs_path=str(tmp_path / "nonexistent"))
        with pytest.raises(ValueError, match="must be an existing directory"):
            ParallelDataFlow.start(flow)

    def test_passes_with_valid_docs_path(self, tmp_path):
        from pipeline import ParallelDataFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")

        flow = self._make_flow(docs_path=str(docs_dir))
        ParallelDataFlow.start(flow)
        assert flow.docs_path_str == str(docs_dir)
        flow.next.assert_called_once()

    def test_defaults_docs_output_to_converted_subdir(self, tmp_path):
        from pipeline import ParallelDataFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")

        flow = self._make_flow(docs_path=str(docs_dir), docs_output="")
        ParallelDataFlow.start(flow)
        assert flow.docs_output_str == str(docs_dir / "converted")

    def test_uses_custom_docs_output(self, tmp_path):
        from pipeline import ParallelDataFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("test\n")
        custom_output = tmp_path / "custom_output"

        flow = self._make_flow(docs_path=str(docs_dir), docs_output=str(custom_output))
        ParallelDataFlow.start(flow)
        assert flow.docs_output_str == str(custom_output)


class TestPipelineConvertDocumentsStep:
    def test_sets_converted_parquet_path(self, tmp_path):
        from pipeline import ParallelDataFlow

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "sample.txt").write_text("Hello world\n")

        output_dir = tmp_path / "output"

        flow = MagicMock(spec=ParallelDataFlow)
        flow.docs_path_str = str(docs_dir)
        flow.docs_output_str = str(output_dir)

        ParallelDataFlow.convert_documents(flow)

        assert hasattr(flow, "converted_parquet_path")
        assert flow.converted_parquet_path.endswith("converted.parquet")
        assert Path(flow.converted_parquet_path).exists()
        flow.next.assert_called_once()
