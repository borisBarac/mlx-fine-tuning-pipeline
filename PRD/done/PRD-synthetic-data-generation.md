# PRD: Synthetic Data Generation

## Problem Statement

Users have converted document text (from PRD-document-conversion) but no labeled training data. Creating QA pairs manually is slow, expensive, and doesn't scale. The pipeline needs a way to automatically generate question-answer training pairs from document text using a teacher model, producing the `instruction`/`output` parquet that the downstream training flow expects.

## Solution

Add a synthetic data generation module that uses a teacher model via the **OpenAI API** (compatible with any OpenAI-compatible endpoint — DeepSeek, Together, OpenAI, etc.) to generate QA pairs from converted document text. The teacher model reads document content chunks and produces question-answer pairs, which are filtered for quality and saved as a parquet file with `instruction` and `output` columns. The generation runs as the `generate_qa` Metaflow pipeline step, between `convert_documents` and `prepare_training`.

## User Stories

1. As a pipeline user, I want to generate QA training pairs from my document text using a teacher model, so that I can create training data without manual labeling.
2. As a pipeline user, I want to choose any OpenAI-compatible teacher model via a `--teacher-model` parameter, so that I can balance cost and quality for my use case.
3. As a pipeline user, I want to control how many QA pairs are generated via a `--num-examples` parameter, so that I can size my dataset to my needs.
4. As a pipeline user, I want to control the chunk size used when splitting documents for the teacher model, so that I can fit within the teacher model's context window.
5. As a pipeline user, I want low-quality QA pairs filtered out automatically, so that my training data is clean without manual review.
6. As a pipeline user, I want to provide my API key via an `--api-key` parameter or the `OPENAI_API_KEY` environment variable, so that authentication is flexible.
7. As a pipeline user, I want to configure the API endpoint via an `--api-base` parameter, so that I can use any OpenAI-compatible provider (DeepSeek, Together, OpenAI, etc.).
8. As a pipeline user, I want generation progress logged per-chunk with counts of successful and failed examples, so that I can monitor long-running generation jobs.
9. As a pipeline developer, I want the generation logic in a new `src/data_prep/generate.py` module, so that it's testable independently of the pipeline.
10. As a pipeline developer, I want conversion errors for individual chunks logged but not fatal, so that one bad chunk doesn't halt generation of the entire dataset.

## Implementation Decisions

- **Teacher model API**: Use the `openai` Python SDK. Any OpenAI-compatible endpoint works — DeepSeek (`https://api.deepseek.com/v1`), Together (`https://api.together.xyz/v1`), OpenAI (`https://api.openai.com/v1`), etc. The SDK's `client.chat.completions.create()` with `response_format={"type": "json_object"}` ensures structured JSON output from compatible models.
- **New module**: `src/data_prep/generate.py` containing:
  - `generate_qa_pairs(content_chunk, client, model, num_examples)` — calls the teacher model and returns a list of QA dicts with `question` and `answer` fields
  - `filter_qa_pairs(examples, min_question_words, min_answer_words)` — quality filter based on word count
  - `generate_dataset(input_parquet, output_dir, client, model, num_examples, chunk_size)` — orchestrates chunking, generation, filtering, and saves a parquet file with `instruction`/`output` columns
- **Chunking strategy**: Text from the conversion step (PRD-document-conversion) is split by `section_type` boundaries first, then by word count (`--generation-chunk-size`, default 2000 words). Each chunk includes a metadata header with source file and page number so the teacher model has context.
- **Example distribution**: `--num-examples` is distributed across chunks proportionally to each chunk's word count. Larger chunks produce more QA pairs. The actual total may differ slightly from requested.
- **Teacher prompt**: The prompt instructs the teacher to generate QA pairs answerable only from the provided content and output as a JSON array. Uses `response_format={"type": "json_object"}`.
- **Quality filtering**: Simple filter based on minimum question length (5 words) and minimum answer length (20 words).
- **Output format**: Parquet file with `instruction` and `output` columns. This matches the existing `transform_parquet_to_jsonl` input format and feeds directly into the downstream training flow via `self.parquet_path_str`.
- **Pipeline flow**: The generation step reads `self.converted_parquet_path` from the conversion step (Metaflow artifact) and sets `self.parquet_path_str` to the generated training parquet for downstream consumption.
- **Pipeline parameters**:
  - `--teacher-model` (string, default `"deepseek-chat"`): Model identifier for the teacher model.
  - `--num-examples` (int, default `200`): Total QA pairs to generate, distributed proportionally across chunks.
  - `--generation-chunk-size` (int, default `2000`): Words per chunk fed to the teacher model.
  - `--api-base` (string, default `"https://api.deepseek.com/v1"`): OpenAI-compatible API endpoint URL.
  - `--api-key` (string, default `""`): API key. Falls back to `OPENAI_API_KEY` environment variable.
- **Error handling**: If a chunk generation fails (API error, invalid JSON, rate limit), log the error and continue with remaining chunks. If fewer than 50% of requested examples are generated, raise a warning. If zero examples are generated, raise an error.
- **Rate limiting**: A fixed 0.5-second delay between API calls to stay within rate limits.
- **Dependency**: Add `openai` to `pyproject.toml` dependencies.

## Testing Decisions

- **Unit tests for `generate_qa_pairs`**: Mock the OpenAI client. Assert the function sends the correct prompt format, handles a valid JSON response, and returns the expected list of dicts with `question` and `answer` fields.
- **Unit tests for `filter_qa_pairs`**: Test with a mix of valid and invalid examples (too short, missing fields). Assert only valid examples pass.
- **Unit tests for `generate_dataset`**: Mock the OpenAI client and use a fixture parquet with `text_content`/`section_type` columns. Assert the output parquet has `instruction` and `output` columns with non-empty values.
- **Error handling tests**: Test with a mocked API that returns invalid JSON, then a timeout, then a rate limit error. Assert errors are logged and partial results are returned.
- **Pipeline integration test**: Test that the generation step reads `self.converted_parquet_path` and sets `self.parquet_path_str` correctly for downstream consumption.

## Out of Scope

- Difficulty levels and chain-of-thought rationale in QA pairs (deferred to V2)
- KD-logit approach — training with soft logits from the teacher model
- DST (Data Selection and joint Training) with student model confidence scoring
- Multi-turn conversation generation
- Streaming responses from the teacher model for real-time progress
- Automatic teacher model selection based on cost or quality benchmarks
- Deduplication of generated QA pairs
- Human-in-the-loop review of generated QA pairs (Prodigy or similar)
- Local teacher model support (Ollama, vLLM)
- Parallel API calls to the teacher model (sequential to respect rate limits)
- Retry logic with exponential backoff (simple fixed delay is sufficient initially)
- Cost tracking for API usage

## Further Notes

- DeepSeek pricing is approximately $0.14/M input tokens and $0.28/M output tokens (DeepSeek-V3). Generating 200 QA pairs from a 50-page document costs roughly $0.01-0.05, making it very cost-effective for batch generation.
- The quality of generated QA pairs depends heavily on the teacher model and the prompt. The prompt is designed to constrain the teacher to only use information present in the source text (no hallucination).
- The generation step is inherently slower than other pipeline steps because it's API-bound. A 200-example generation with DeepSeek takes roughly 2-5 minutes depending on document length. The 0.5-second delay between API calls prevents rate limiting but adds to total time.
- The generated dataset should be reviewed by a human before production use. A spot-check of 10-20 examples is usually sufficient to catch systematic issues (repetitive questions, off-topic answers, formatting problems).
- This step always runs as part of the linear pipeline, consuming the converted parquet from the `convert_documents` step via Metaflow artifact passing.
