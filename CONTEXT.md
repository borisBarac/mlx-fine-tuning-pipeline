# MLX Fine-Tuning Pipeline

A Metaflow pipeline that converts raw documents into fine-tuned language models. The pipeline runs locally on Apple Silicon using MLX.

## Language

**Document Conversion**:
Extracting structured text (markdown) from raw document files (PDF, DOCX, PPTX, TXT) using Docling.
_Avoid_: text extraction, parsing, ingestion

**Synthetic Data Generation**:
Generating question-answer training pairs from converted document text using a teacher model via an OpenAI-compatible API.
_Avoid_: data generation, KD-SDG (use only when referencing the specific technique)

**Teacher Model**:
A large language model (accessed via an OpenAI-compatible API) that reads document content and produces QA pairs.
_Avoid_: generator model, LLM

**Converted Parquet**:
The parquet file produced by document conversion, with columns `source_file`, `page_number`, `section_type`, `text_content`.
_Avoid_: conversion output, document parquet

**Training Parquet**:
The parquet file with `instruction` and `output` columns that feeds the training flow.
_Avoid_: QA parquet, final parquet

**Section Type**:
A label for a structural element in a document: `text`, `title`, `section_header`, `table`, `list_item`, `code`, `formula`.
_Avoid_: element type, block type
