import json
import time
from pathlib import Path

import polars as pl
import pytest

from src.data_prep.convert import convert_documents

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestConvertDocumentsHappyPath:
    def test_converts_txt_file_to_parquet(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "sample.txt").write_text(
            "Hello world\n\nThis is a test.\n", encoding="utf-8"
        )

        output = tmp_path / "output"
        parquet_path = convert_documents(str(source), str(output))

        assert Path(parquet_path).exists()
        df = pl.read_parquet(parquet_path)
        assert set(df.columns) == {
            "source_file",
            "page_number",
            "section_type",
            "text_content",
        }
        assert len(df) > 0
        assert all(row["text_content"] != "" for row in df.iter_rows(named=True))

    def test_saves_markdown_per_document(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "sample.txt").write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        convert_documents(str(source), str(output))

        md_files = list(output.glob("*.md"))
        assert len(md_files) == 1
        assert md_files[0].name == "sample.txt.md"
        assert md_files[0].read_text(encoding="utf-8").strip() != ""

    def test_writes_conversion_report(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "sample.txt").write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        convert_documents(str(source), str(output))

        report_path = output / "conversion_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert "sample.txt" in report["successes"]
        assert report["failures"] == []

    def test_section_types_are_valid(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "sample.txt").write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        convert_documents(str(source), str(output))

        df = pl.read_parquet(output / "converted.parquet")
        valid_types = {
            "text",
            "title",
            "section_header",
            "table",
            "list_item",
            "code",
            "formula",
        }
        for row in df.iter_rows(named=True):
            assert row["section_type"] in valid_types

    def test_output_parquet_columns(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "sample.txt").write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        convert_documents(str(source), str(output))

        df = pl.read_parquet(output / "converted.parquet")
        assert df.schema == {
            "source_file": pl.String,
            "page_number": pl.Int64,
            "section_type": pl.String,
            "text_content": pl.String,
        }


class TestConvertDocumentsErrorHandling:
    def test_corrupted_file_logged_not_fatal(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "corrupted.pdf").write_bytes(b"not a real pdf content here")
        (source / "good.txt").write_text("This is valid text.\n", encoding="utf-8")

        output = tmp_path / "output"
        parquet_path = convert_documents(str(source), str(output))

        df = pl.read_parquet(parquet_path)
        assert len(df) > 0

        report = json.loads(
            (output / "conversion_report.json").read_text(encoding="utf-8")
        )
        assert "good.txt" in report["successes"]
        assert len(report["failures"]) >= 1
        assert any(f["file"] == "corrupted.pdf" for f in report["failures"])

    def test_all_failures_raises(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "bad.pdf").write_bytes(b"not a real pdf")

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="failed to convert"):
            convert_documents(str(source), str(output))

    def test_no_supported_documents_raises(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()

        output = tmp_path / "output"
        with pytest.raises(RuntimeError, match="No supported documents"):
            convert_documents(str(source), str(output))


class TestConvertDocumentsSkippedFormats:
    def test_unsupported_extension_skipped(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        (source / "audio.wav").write_bytes(b"fake audio data")
        (source / "good.txt").write_text("Valid text.\n", encoding="utf-8")

        output = tmp_path / "output"
        parquet_path = convert_documents(str(source), str(output))

        df = pl.read_parquet(parquet_path)
        assert len(df) > 0

        report = json.loads(
            (output / "conversion_report.json").read_text(encoding="utf-8")
        )
        assert "audio.wav" in report["skipped"]
        assert "good.txt" in report["successes"]


class TestConvertDocumentsCaching:
    def test_skips_conversion_when_up_to_date(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        txt_file = source / "sample.txt"
        txt_file.write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        parquet_path = convert_documents(str(source), str(output))

        first_mtime = Path(parquet_path).stat().st_mtime
        time.sleep(0.1)

        parquet_path_2 = convert_documents(str(source), str(output))
        assert parquet_path == parquet_path_2
        assert Path(parquet_path).stat().st_mtime == first_mtime

    def test_reconverts_when_source_modified(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        txt_file = source / "sample.txt"
        txt_file.write_text("Hello world\n", encoding="utf-8")

        output = tmp_path / "output"
        parquet_path = convert_documents(str(source), str(output))
        first_mtime = Path(parquet_path).stat().st_mtime

        time.sleep(0.1)
        txt_file.write_text("Modified content\n", encoding="utf-8")

        parquet_path_2 = convert_documents(str(source), str(output))
        assert Path(parquet_path_2).stat().st_mtime > first_mtime
