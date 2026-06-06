# PRD: Document-to-Text Conversion with Docling

## Problem Statement

Users who have raw documents — PDFs, Word files, slide decks, plain text files — must manually extract and format text before using the pipeline. This is a manual, error-prone step that involves dealing with page layouts, tables, and reading order. Without a conversion step, the pipeline cannot ingest the most common document formats that organizations actually produce and store.

## Solution

Add a document conversion module powered by **Docling** (IBM's open-source document processing library) that converts PDF, DOCX, PPTX, and plain text files into structured raw text. This runs as the first Metaflow pipeline step (`convert_documents`) in a single linear pipeline path. The step outputs a parquet file with columns `source_file`, `page_number`, `section_type`, and `text_content`, and saves one markdown file per source document to the output directory. The parquet is passed as a Metaflow artifact to the synthetic data generation step (PRD-synthetic-data-generation), which produces QA pairs and feeds the rest of the training pipeline.

The full pipeline is: `start → convert_documents → generate_qa → prepare_training → process_chunks → merge_results → train_model → end`.

## User Stories

1. As a pipeline user, I want to point the pipeline at a directory of PDF files and have them converted to structured text automatically, so that I don't need to manually extract text from each file.
2. As a pipeline user, I want to convert DOCX, PPTX, and TXT files alongside PDFs through the same interface, so that I can handle mixed document collections without different tools.
3. As a pipeline user, I want tables preserved as structured data (not lost or mangled), so that tabular information is available for downstream QA generation.
4. As a pipeline user, I want section types (heading, paragraph, table, list) identified in the output, so that downstream steps can use document structure when generating training examples.
5. As a pipeline user, I want the conversion step to run locally without sending documents to external APIs, so that sensitive documents stay private.
6. As a pipeline user, I want to specify a directory of files via a `--docs-path` parameter, so that I can batch-process document collections.
7. As a pipeline user, I want the output parquet file to be cached and reused on subsequent runs when the source documents haven't changed, so that I avoid redundant conversion.
8. As a pipeline user, I want markdown files saved alongside the parquet output for each converted document, so that I can inspect the extracted text in a human-readable format.
9. As a pipeline developer, I want the conversion logic in a new `src/data_prep/convert.py` module, so that it's testable independently of the pipeline.
10. As a pipeline developer, I want conversion errors for individual files logged but not fatal, so that one bad file doesn't halt processing of an entire batch.

## Implementation Decisions

- **Library**: Use `docling` (PyPI package `docling`, MIT license). Docling provides layout analysis, table structure recognition (via TableFormer), and multi-format parsing in a single library. It runs locally on CPU at 1-3 pages/second, which is sufficient for batch preprocessing.
- **Supported formats**: PDF, DOCX, PPTX, and plain text (TXT). Files with other extensions are skipped with a warning.
- **New module**: `src/data_prep/convert.py` containing a single function: `convert_documents(source_dir, output_dir)` that handles all files in the directory.
- **Docling usage**: Use `DocumentConverter` for default conversion and `export_to_markdown()` for output. Use `PdfFormatOption` with `InputFormat.PDF` for PDF-specific configuration.
- **Output schema**: Parquet file with columns:
  - `source_file` (string): Original filename
  - `page_number` (int): Page number (null for non-paginated formats)
  - `section_type` (string): One of `text`, `title`, `section_header`, `table`, `list_item`, `code`, `formula`
  - `text_content` (string): Extracted text content as markdown
- **Markdown files**: One `.md` file per source document is saved to the output directory (e.g., `report.pdf.md`). These are always generated alongside the parquet.
- **Pipeline flow**: The pipeline is a single linear path. `start()` validates `--docs-path` and passes to `convert_documents`. The conversion step sets `self.converted_parquet_path` as a Metaflow artifact. The generation step reads this artifact and produces a parquet with `instruction`/`output` columns, which feeds the existing training flow.
- **Pipeline parameters**:
  - `--docs-path` (string, required): Path to a directory of documents.
  - `--docs-output` (string, default `""`): Directory where converted parquet, markdown files, and conversion report are written. Defaults to a `converted/` subdirectory inside `--docs-path`.
- **Caching**: If the output parquet already exists and no source documents have been modified since it was created, skip conversion. Implemented by comparing the output parquet modification time against the latest source document modification time.
- **Error handling**: Failed conversions are logged with the source filename and error message. A `conversion_report.json` is written to the output directory summarizing successes and failures for human inspection. The pipeline only raises if zero documents convert successfully.
- **Dependency**: Add `docling` to `pyproject.toml` dependencies.
- **Removed parameters**: `--parquet-path` is removed. `--docs-format` is removed (markdown is the only format). `--docs-output` defaults relative to `--docs-path` instead of a hardcoded path.

## Testing Decisions

- **Unit tests for `convert.py`**: Test `convert_documents` with a directory containing a small fixture PDF and TXT file (checked into `test/fixtures/`). Assert the output parquet has the expected columns and at least one row with non-empty `text_content`. Assert one `.md` file per source document is saved.
- **Error handling tests**: Test with a corrupted PDF fixture alongside a valid TXT file. Assert the error is logged, the report records the failure, and the function returns results for the valid file.
- **Skipped format tests**: Test with an unsupported file extension (e.g., `.wav`). Assert the file is skipped with a warning and not counted as a failure.
- **Caching tests**: Test that a second call with unchanged source files skips conversion. Test that modifying a source file triggers reconversion.
- **Pipeline integration test**: Test that `start()` validates `--docs-path` exists and passes it to `convert_documents`. Test that the conversion step sets `self.converted_parquet_path` correctly.

## Out of Scope

- OCR for scanned PDFs or images
- XLSX, HTML, and image file conversion
- spaCy NLP annotations on extracted text (POS tags, NER, dependencies)
- Visual Language Model (VLM) based conversion via GraniteDocling
- Layout feature extraction (bounding boxes, reading order coordinates) beyond section type
- Image extraction from documents
- Audio/video file conversion
- LaTeX, XBRL, USPTO, JATS XML schema support
- Cloud-based OCR or remote document processing
- Prodigy annotation integration
- Deduplication of extracted text across documents
- Chunking of extracted text (handled by downstream PRD-synthetic-data-generation)

## Further Notes

- Docling processes documents locally on CPU at approximately 1-3 pages/second (native backend) or 2-3 pages/second (pypdfium backend). A 100-page document takes roughly 30-50 seconds. This is acceptable for batch preprocessing but means conversion should not be re-run unnecessarily — hence the caching strategy.
- The output parquet schema is designed to be consumed by the synthetic data generation flow (PRD-synthetic-data-generation), which will chunk `text_content` by `section_type` and `page_number` to create QA pairs with context about where in the document the answer originated.
- Docling's TableFormer model preserves table structure as Markdown tables in the text output. This is the recommended representation for downstream LLM consumption.
- The `--docs-output` path defaults to a `converted/` subdirectory inside `--docs-path`, keeping output alongside source documents.
