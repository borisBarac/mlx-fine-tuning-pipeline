# Checklist: Synthetic Data Generation

- [x] Add `openai` to pyproject.toml dependencies
- [x] Create `src/data_prep/generate.py` with `generate_qa_pairs`, `filter_qa_pairs`, `generate_dataset`
- [x] Write tests for `generate_qa_pairs` (mocked OpenAI client)
- [x] Write tests for `filter_qa_pairs`
- [x] Write tests for `generate_dataset` (mocked client, fixture parquet)
- [x] Write error handling tests (invalid JSON, timeout, rate limit)
- [x] Update pipeline.py `generate_qa` step with parameters and implementation
- [x] Write pipeline integration test for `generate_qa` step
- [x] All tests pass, lint clean
