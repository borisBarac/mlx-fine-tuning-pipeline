import json
import logging
from pathlib import Path

import polars as pl
from docling_core.types.doc.labels import DocItemLabel
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt"}

LABEL_TO_SECTION_TYPE = {
    DocItemLabel.TITLE: "title",
    DocItemLabel.SECTION_HEADER: "section_header",
    DocItemLabel.PARAGRAPH: "text",
    DocItemLabel.TEXT: "text",
    DocItemLabel.TABLE: "table",
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.CODE: "code",
    DocItemLabel.FORMULA: "formula",
}

OUTPUT_PARQUET_NAME = "converted.parquet"
REPORT_NAME = "conversion_report.json"


def _should_skip_conversion(source_dir: Path, output_dir: Path) -> bool:
    output_parquet = output_dir / OUTPUT_PARQUET_NAME
    if not output_parquet.exists():
        return False

    output_mtime = output_parquet.stat().st_mtime
    return not any(
        f.stat().st_mtime > output_mtime
        for f in source_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _extract_items(doc, source_filename: str) -> list[dict]:
    rows = []
    for item, _level in doc.iterate_items():
        label = item.label
        section_type = LABEL_TO_SECTION_TYPE.get(label, "text")

        text_content = ""
        if hasattr(item, "text"):
            text_content = item.text or ""
        elif hasattr(item, "export_to_markdown"):
            text_content = item.export_to_markdown(doc) or ""

        if not text_content.strip():
            continue

        page_number = None
        if hasattr(item, "prov") and item.prov:
            page_number = item.prov[0].page_no

        rows.append(
            {
                "source_file": source_filename,
                "page_number": page_number,
                "section_type": section_type,
                "text_content": text_content,
            }
        )
    return rows


def convert_documents(source_dir: str, output_dir: str) -> str:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if _should_skip_conversion(source_path, output_path):
        logger.info("Skipping conversion: output is up-to-date")
        return str(output_path / OUTPUT_PARQUET_NAME)

    converter = DocumentConverter()
    all_rows: list[dict] = []
    successes: list[str] = []
    failures: list[dict] = []
    skipped: list[str] = []

    source_files = sorted(f for f in source_path.iterdir() if f.is_file())

    for source_file in source_files:
        ext = source_file.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skipped.append(source_file.name)
            logger.warning("Skipping unsupported file: %s", source_file.name)
            continue

        try:
            result = converter.convert(str(source_file), raises_on_error=False)
            if result.status.value != "success":
                raise RuntimeError(
                    f"Conversion failed with status: {result.status.value}"
                )

            markdown_text = result.document.export_to_markdown()
            md_path = output_path / f"{source_file.name}.md"
            md_path.write_text(markdown_text, encoding="utf-8")

            rows = _extract_items(result.document, source_file.name)
            all_rows.extend(rows)
            successes.append(source_file.name)
            logger.info("Converted: %s (%d sections)", source_file.name, len(rows))

        except Exception as e:
            failures.append({"file": source_file.name, "error": str(e)})
            logger.error("Failed to convert %s: %s", source_file.name, e)

    report = {
        "successes": successes,
        "failures": failures,
        "skipped": skipped,
    }
    report_path = output_path / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not all_rows:
        if failures:
            raise RuntimeError(
                f"All {len(failures)} document(s) failed to convert. "
                f"See {report_path} for details."
            )
        raise RuntimeError(
            f"No supported documents found in {source_dir}. "
            f"See {report_path} for details."
        )

    df = pl.DataFrame(
        all_rows,
        schema={
            "source_file": pl.String,
            "page_number": pl.Int64,
            "section_type": pl.String,
            "text_content": pl.String,
        },
    )

    parquet_path = output_path / OUTPUT_PARQUET_NAME
    df.write_parquet(parquet_path)
    logger.info("Wrote %d rows to %s", len(df), parquet_path)

    return str(parquet_path)
