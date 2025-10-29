```console

f1_pipeline is 📦 v0.1.0 via 🐍 v3.12.0 (f1-pipeline) on ☁️  (eu-central-1)
❯ uv run python ./src/pipeline_ml.py run --parquet-path ./LLM/parquet_sets/train-00000-of-00001.parquet
Metaflow 2.19.3 executing ParallelDataFlow for user:boris
Validating your flow...
    The graph looks good!
Running pylint...
    Pylint not found, so extra checks are disabled.
2025-10-30 17:03:47.123 Workflow starting (run-id 1761840227122170):
2025-10-30 17:03:47.137 [1761840227122170/start/1 (pid 79224)] Task is starting.
2025-10-30 17:03:48.250 [1761840227122170/start/1 (pid 79224)] Starting parallel data processing
2025-10-30 17:03:48.278 [1761840227122170/start/1 (pid 79224)] Input: ./LLM/parquet_sets/train-00000-of-00001.parquet
2025-10-30 17:03:48.278 [1761840227122170/start/1 (pid 79224)] Output: /Users/boris/Desktop/f1_pipeline/LLM/data
2025-10-30 17:03:48.278 [1761840227122170/start/1 (pid 79224)] Chunk size: 1000 rows
2025-10-30 17:03:48.278 [1761840227122170/start/1 (pid 79224)] Dataset contains 507 rows (metadata loaded in 0.03s)
2025-10-30 17:03:48.352 [1761840227122170/start/1 (pid 79224)] Processing 1 chunks in parallel
2025-10-30 17:03:48.354 [1761840227122170/start/1 (pid 79224)] Task finished successfully.
2025-10-30 17:03:48.360 [1761840227122170/process_chunks/2 (pid 79231)] Task is starting.
2025-10-30 17:03:49.451 [1761840227122170/process_chunks/2 (pid 79231)] Processing 1 chunks in parallel...
2025-10-30 17:03:49.451 [1761840227122170/process_chunks/2 (pid 79231)] Using temporary directory: /var/folders/xp/2fyz9z3j7mgfzfkdtpcn95kr0000gn/T/parallel_chunks_1ua35dt_
2025-10-30 17:03:49.559 [1761840227122170/process_chunks/2 (pid 79231)] Parallel processing completed in 0.11s
2025-10-30 17:03:49.635 [1761840227122170/process_chunks/2 (pid 79231)] Successful chunks: 1/1
2025-10-30 17:03:49.636 [1761840227122170/process_chunks/2 (pid 79231)] Task finished successfully.
2025-10-30 17:03:49.643 [1761840227122170/merge_results/3 (pid 79239)] Task is starting.
2025-10-30 17:03:50.715 [1761840227122170/merge_results/3 (pid 79239)] Merging processed chunks...
2025-10-30 17:03:50.716 [1761840227122170/merge_results/3 (pid 79239)] Merged 1 files with 507 total lines
2025-10-30 17:03:50.720 [1761840227122170/merge_results/3 (pid 79239)] Splitting 507 rows: 481 train, 26 validation
2025-10-30 17:03:50.725 [1761840227122170/merge_results/3 (pid 79239)] Merge completed in 0.01s
2025-10-30 17:03:50.798 [1761840227122170/merge_results/3 (pid 79239)] Final files created:
2025-10-30 17:03:50.798 [1761840227122170/merge_results/3 (pid 79239)] Train: /Users/boris/Desktop/f1_pipeline/LLM/data/train.jsonl (481 samples)
2025-10-30 17:03:50.798 [1761840227122170/merge_results/3 (pid 79239)] Valid: /Users/boris/Desktop/f1_pipeline/LLM/data/valid.jsonl (26 samples)
2025-10-30 17:03:50.799 [1761840227122170/merge_results/3 (pid 79239)] Task finished successfully.
2025-10-30 17:03:50.805 [1761840227122170/train_model/4 (pid 79246)] Task is starting.
2025-10-30 17:03:51.879 [1761840227122170/train_model/4 (pid 79246)] Starting model training...
2025-10-30 17:03:51.880 [1761840227122170/train_model/4 (pid 79246)] Training data path: /Users/boris/Desktop/f1_pipeline/LLM/data
2025-10-30 17:03:51.880 [1761840227122170/train_model/4 (pid 79246)] Training iterations: 100
2025-10-30 17:03:51.880 [1761840227122170/train_model/4 (pid 79246)] Batch size: 4
2025-10-30 17:03:52.030 [1761840227122170/train_model/4 (pid 79246)] Fetching 9 files:   0%|          | 0/9 [00:Fetching 9 files: 100%|██████████| 9/9 [00:00<00:00, 155344.59it/s]
2025-10-30 17:03:52.843 [1761840227122170/train_model/4 (pid 79246)] Model and tokenizer loaded successfully.
2025-10-30 17:03:52.855 [1761840227122170/train_model/4 (pid 79246)] Training set size: 481
2025-10-30 17:03:52.856 [1761840227122170/train_model/4 (pid 79246)] Validation set size: 26
2025-10-30 17:03:52.856 [1761840227122170/train_model/4 (pid 79246)] Total iterations: 100
2025-10-30 17:03:52.856 [1761840227122170/train_model/4 (pid 79246)] Training started...
2025-10-30 17:03:52.856 [1761840227122170/train_model/4 (pid 79246)] Trainable parameters: 0.249% (4.063M/1631.750M)
2025-10-30 17:03:52.856 [1761840227122170/train_model/4 (pid 79246)] Starting training..., iters: 100
2025-10-30 17:03:55.629 [1761840227122170/train_model/4 (pid 79246)] Calculating loss...:   0%|          | 0/6 [Calculating loss...: 100%|██████████| 6/6 [00:02<00:00,  2.22it/s]
2025-10-30 17:03:55.631 [1761840227122170/train_model/4 (pid 79246)] Iter 1: Val loss 2.699, Val took 2.702s
2025-10-30 17:04:03.375 [1761840227122170/train_model/4 (pid 79246)] Iter 10: Train loss 2.198, Learning Rate 1.000e-05, It/sec 1.293, Tokens/sec 267.606, Trained Tokens 2070, Peak mem 2.283 GB
2025-10-30 17:04:10.078 [1761840227122170/train_model/4 (pid 79246)] Iter 20: Train loss 1.299, Learning Rate 1.000e-05, It/sec 1.493, Tokens/sec 291.196, Trained Tokens 4020, Peak mem 2.283 GB
2025-10-30 17:04:16.788 [1761840227122170/train_model/4 (pid 79246)] Iter 30: Train loss 1.248, Learning Rate 1.000e-05, It/sec 1.492, Tokens/sec 253.876, Trained Tokens 5722, Peak mem 2.283 GB
2025-10-30 17:04:24.188 [1761840227122170/train_model/4 (pid 79246)] Iter 40: Train loss 1.156, Learning Rate 1.000e-05, It/sec 1.354, Tokens/sec 257.040, Trained Tokens 7621, Peak mem 2.695 GB
2025-10-30 17:04:31.185 [1761840227122170/train_model/4 (pid 79246)] Iter 50: Train loss 0.814, Learning Rate 1.000e-05, It/sec 1.430, Tokens/sec 259.764, Trained Tokens 9437, Peak mem 2.695 GB
2025-10-30 17:04:38.423 [1761840227122170/train_model/4 (pid 79246)] Iter 60: Train loss 0.739, Learning Rate 1.000e-05, It/sec 1.384, Tokens/sec 252.130, Trained Tokens 11259, Peak mem 2.695 GB
2025-10-30 17:04:45.311 [1761840227122170/train_model/4 (pid 79246)] Iter 70: Train loss 0.642, Learning Rate 1.000e-05, It/sec 1.454, Tokens/sec 271.740, Trained Tokens 13128, Peak mem 2.695 GB
2025-10-30 17:04:52.822 [1761840227122170/train_model/4 (pid 79246)] Iter 80: Train loss 0.641, Learning Rate 1.000e-05, It/sec 1.334, Tokens/sec 267.059, Trained Tokens 15130, Peak mem 2.695 GB
2025-10-30 17:04:59.875 [1761840227122170/train_model/4 (pid 79246)] Iter 90: Train loss 0.578, Learning Rate 1.000e-05, It/sec 1.419, Tokens/sec 251.277, Trained Tokens 16901, Peak mem 2.695 GB
2025-10-30 17:05:08.272 [1761840227122170/train_model/4 (pid 79246)] Calculating loss...:   0%|          | 0/6 [Calculating loss...: 100%|██████████| 6/6 [00:02<00:00,  2.47it/s]
2025-10-30 17:05:08.274 [1761840227122170/train_model/4 (pid 79246)] Iter 100: Val loss 0.922, Val took 2.440s
2025-10-30 17:05:08.914 [1761840227122170/train_model/4 (pid 79246)] Iter 100: Train loss 0.503, Learning Rate 1.000e-05, It/sec 1.517, Tokens/sec 268.851, Trained Tokens 18673, Peak mem 2.695 GB
2025-10-30 17:05:08.930 [1761840227122170/train_model/4 (pid 79246)] Iter 100: Saved adapter weights to adapters/adapters.safetensors and adapters/0000100_adapters.safetensors.
2025-10-30 17:05:08.937 [1761840227122170/train_model/4 (pid 79246)] Saved final weights to adapters/adapters.safetensors.
2025-10-30 17:05:08.972 [1761840227122170/train_model/4 (pid 79246)] Training completed.
2025-10-30 17:05:08.980 [1761840227122170/train_model/4 (pid 79246)] Model training completed in 77.10s
2025-10-30 17:05:09.214 [1761840227122170/train_model/4 (pid 79246)] Task finished successfully.
2025-10-30 17:05:09.220 [1761840227122170/end/5 (pid 79277)] Task is starting.
2025-10-30 17:05:10.882 [1761840227122170/end/5 (pid 79277)] Cleaned up temporary directory: /var/folders/xp/2fyz9z3j7mgfzfkdtpcn95kr0000gn/T/parallel_chunks_1ua35dt_
2025-10-30 17:05:10.882 [1761840227122170/end/5 (pid 79277)]
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] ============================================================
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] PARALLEL DATA PROCESSING COMPLETED
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] ============================================================
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] Total rows: 507 | Chunks: 1
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] Train: 481 (94.9%)
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] Valid: 26 (5.1%)
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)]
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] Timing:
2025-10-30 17:05:10.883 [1761840227122170/end/5 (pid 79277)] Metadata: 0.03s | Parallel: 0.11s
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)] Merge: 0.01s | Training: 77.10s
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)] Total: 77.25s
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)]
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)] Output:
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)] /Users/boris/Desktop/f1_pipeline/LLM/data/train.jsonl
2025-10-30 17:05:10.884 [1761840227122170/end/5 (pid 79277)] /Users/boris/Desktop/f1_pipeline/LLM/data/valid.jsonl
2025-10-30 17:05:10.960 [1761840227122170/end/5 (pid 79277)] ============================================================
2025-10-30 17:05:10.961 [1761840227122170/end/5 (pid 79277)] Task finished successfully.
2025-10-30 17:05:10.961 Done!

```