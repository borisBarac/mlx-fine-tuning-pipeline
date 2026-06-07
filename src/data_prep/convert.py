import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from docling_core.types.doc.labels import DocItemLabel
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

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


def _build_converter(num_threads: int) -> DocumentConverter:
    accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=AcceleratorDevice.CPU,
    )

    pdf_pipeline = ThreadedPdfPipelineOptions(
        accelerator_options=accelerator_options,
        do_ocr=False,
        layout_batch_size=32,
        table_batch_size=4,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: FormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=DoclingParseDocumentBackend,
                pipeline_options=pdf_pipeline,
            ),
        }
    )


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


def _convert_single_file(args: tuple[str, int]) -> dict:
    source_file_str, num_threads = args
    source_file = Path(source_file_str)
    ext = source_file.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "filename": source_file.name,
            "status": "skipped",
            "rows": [],
            "markdown_text": None,
            "error": None,
        }

    converter = _build_converter(num_threads)

    try:
        result = converter.convert(source_file_str, raises_on_error=False)
        if result.status.value != "success":
            raise RuntimeError(f"Conversion failed with status: {result.status.value}")

        markdown_text = result.document.export_to_markdown()
        rows = _extract_items(result.document, source_file.name)

        return {
            "filename": source_file.name,
            "status": "success",
            "rows": rows,
            "markdown_text": markdown_text,
            "error": None,
        }
    except Exception as e:
        return {
            "filename": source_file.name,
            "status": "failure",
            "rows": [],
            "markdown_text": None,
            "error": str(e),
        }


def convert_documents(
    source_dir: str,
    output_dir: str,
    num_threads: int = 4,
) -> str:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["DOCLING_NUM_THREADS"] = str(num_threads)

    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if _should_skip_conversion(source_path, output_path):
        logger.info("Skipping conversion: output is up-to-date")
        return str(output_path / OUTPUT_PARQUET_NAME)

    source_files = sorted(f for f in source_path.iterdir() if f.is_file())
    convertible = [f for f in source_files if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    max_workers = max(1, (os.cpu_count() or 4) // num_threads)

    if max_workers > 1 and len(convertible) > 1:
        file_args = [(str(f), num_threads) for f in convertible]
        logger.info(
            "Converting %d files with %d workers, %d threads each",
            len(convertible),
            max_workers,
            num_threads,
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_convert_single_file, file_args))
    else:
        logger.info(
            "Converting %d files sequentially, %d threads",
            len(convertible),
            num_threads,
        )
        results = [_convert_single_file((str(f), num_threads)) for f in convertible]

    all_rows: list[dict] = []
    successes: list[str] = []
    failures: list[dict] = []
    skipped: list[str] = []

    skipped_names = {f.name for f in source_files} - {f.name for f in convertible}
    skipped.extend(sorted(skipped_names))
    for name in skipped:
        logger.warning("Skipping unsupported file: %s", name)

    for result in results:
        filename = result["filename"]
        status = result["status"]

        if status == "skipped":
            skipped.append(filename)
            logger.warning("Skipping unsupported file: %s", filename)
        elif status == "success":
            md_text = result["markdown_text"]
            if md_text is not None:
                md_path = output_path / f"{filename}.md"
                md_path.write_text(md_text, encoding="utf-8")

            all_rows.extend(result["rows"])
            successes.append(filename)
            logger.info("Converted: %s (%d sections)", filename, len(result["rows"]))
        elif status == "failure":
            failures.append({"file": filename, "error": result["error"]})
            logger.error("Failed to convert %s: %s", filename, result["error"])

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
