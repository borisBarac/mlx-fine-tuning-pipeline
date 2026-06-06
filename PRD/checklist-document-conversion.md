# Checklist: PRD-document-conversion

- [x] Add `docling` to pyproject.toml dependencies
- [x] Create `src/data_prep/convert.py` with `convert_documents(source_dir, output_dir)` function
  - Supports PDF, DOCX, PPTX, TXT formats
  - Outputs parquet with columns: source_file, page_number, section_type, text_content
  - Saves one .md file per source document
  - Caching: skip conversion if output parquet exists and is newer than all source files
  - Error handling: log failures per file, write conversion_report.json, raise only if zero successes
- [x] Update `src/pipeline.py` with `convert_documents` step and `--docs-path`/`--docs-output` parameters
- [x] Create test fixtures in `test/fixtures/` (TXT file, minimal test data)
- [x] Write unit tests for convert.py
  - Happy path with TXT file
  - Error handling with corrupted file
  - Skipped unsupported format
  - Caching (skip when up-to-date, reconvert when source modified)
- [x] Write pipeline integration tests (start step validates docs-path)
- [x] Run all tests and linters
