# Checklist: Regenerate README.md

- [x] Fix project description: general-purpose fine-tuning pipeline, not just a tutorial
- [x] Update architecture section to reflect 3 pipelines (data_prep_flow, training_flow, pipeline)
- [x] Update data_prep modules: convert.py (Docling), generate.py (teacher model QA), transform.py
- [x] Fix install command from `uv pip install -r pylock.toml` to `uv sync`
- [x] Replace F1-specific usage examples with document-based pipeline usage
- [x] Update API key setup to reflect OPENROUTER_API_KEY / OPENAI_API_KEY env vars
- [x] Remove references to non-existent `pipeline_ml.py` (now `pipeline.py`)
- [x] Fix pipeline parameters to match current CLI flags (--docs-path, --teacher-model, etc.)
- [x] Add training pipeline parameters (--model, --backend, --lora-rank, etc.)
- [x] Fix test command to `uv run pytest`
- [x] Remove outdated Metaflow dev services section
- [x] Update technology stack (add Docling, OpenAI-compatible API)
- [x] Remove pipline_output_example.md reference and typo
