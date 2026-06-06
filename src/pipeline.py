import os
import shutil
import tempfile
from pathlib import Path

from metaflow import Parameter, parallel_map, resources, step  # type: ignore
from metaflow.flowspec import FlowSpec

from data_prep.convert import convert_documents
from data_prep.create_training_data import merge_jsonl_files
from data_prep.transform import transform_parquet_to_jsonl
from training.trainer import train_model, export_model


class ParallelDataFlow(FlowSpec):
    script_dir = Path(__file__).parent.resolve()

    docs_path = Parameter(
        "docs-path",
        help="Path to a directory of documents to convert",
        default="",
    )

    docs_output = Parameter(
        "docs-output",
        help="Directory for converted output. Defaults to converted/ inside --docs-path",
        default="",
    )

    chunk_size = Parameter(
        "chunk-size",
        help="Number of rows per chunk for parallel processing",
        default=100 if os.getenv("TEST_MODE") == "true" else 1000,
        type=int,
    )

    training_iters = Parameter(
        "training-iters",
        help="Number of training iterations",
        default=100,
        type=int,
    )

    batch_size = Parameter(
        "batch-size",
        help="Batch size for training",
        default=4,
        type=int,
    )

    backend = Parameter(
        "backend",
        help="Training backend: mlx or cuda",
        default="mlx",
    )

    model_name = Parameter(
        "model",
        help="HuggingFace model identifier",
        default="unsloth/granite-4.0-1b",
    )

    chat_template = Parameter(
        "chat-template",
        help="Chat template name for Unsloth",
        default="llama-3.1",
    )

    learning_rate = Parameter(
        "learning-rate",
        help="Learning rate for training",
        default=2e-4,
        type=float,
    )

    lora_rank = Parameter(
        "lora-rank",
        help="LoRA rank for fine-tuning",
        default=64,
        type=int,
    )

    max_seq_length = Parameter(
        "max-seq-length",
        help="Maximum sequence length",
        default=2048,
        type=int,
    )

    export_format = Parameter(
        "export-format",
        help="Export format: none or safetensors",
        default="none",
    )

    num_threads = Parameter(
        "num-threads",
        help="Number of threads per document converter",
        default=4,
        type=int,
    )

    @resources(memory=1000, cpu=1)
    @step
    def start(self):
        """Initialize pipeline: validate docs-path and begin conversion"""
        self.docs_path_str: str = str(self.docs_path)
        if not self.docs_path_str:
            raise ValueError("--docs-path is required")

        if not Path(self.docs_path_str).is_dir():
            raise ValueError(
                f"--docs-path must be an existing directory: {self.docs_path_str}"
            )

        self.output_dir_str: str = str(self.script_dir.parent / "LLM" / "data")
        self.chunk_size_int: int = int(self.chunk_size)  # type: ignore

        if self.docs_output:
            self.docs_output_str: str = str(self.docs_output)
        else:
            self.docs_output_str: str = str(Path(self.docs_path_str) / "converted")

        print("Starting pipeline")
        print(f"Documents: {self.docs_path_str}")
        print(f"Output: {self.docs_output_str}")

        self.next(self.convert_documents)

    @resources(memory=4000, cpu=8)
    @step
    def convert_documents(self):
        """Convert documents to structured text parquet"""
        print("Converting documents...")

        self.converted_parquet_path: str = convert_documents(
            self.docs_path_str,
            self.docs_output_str,
            num_threads=int(self.num_threads),
        )

        print(f"Converted parquet: {self.converted_parquet_path}")
        self.next(self.generate_qa)

    @resources(memory=1000, cpu=1)
    @step
    def generate_qa(self):
        """Generate QA pairs from converted documents (placeholder)"""
        raise NotImplementedError(
            "generate_qa step not yet implemented. See PRD-synthetic-data-generation."
        )

    @resources(memory=1000, cpu=2)
    @step
    def process_chunks(self):
        """Process data chunks in parallel using parallel_map"""
        import time

        print(f"Processing chunks from: {self.parquet_path_str}")

        self.metadata_time = 0.0

        self.num_chunks = (
            self.total_rows + self.chunk_size_int - 1
        ) // self.chunk_size_int
        self.chunks = []
        for i in range(self.num_chunks):
            start_row = i * self.chunk_size_int
            end_row = min((i + 1) * self.chunk_size_int, self.total_rows)
            self.chunks.append((start_row, end_row))

        print(f"Processing {self.num_chunks} chunks in parallel")

        self.temp_dir = tempfile.mkdtemp(prefix="parallel_chunks_")
        print(f"Using temporary directory: {self.temp_dir}")

        def process_chunk(chunk_info):
            """Process a single chunk of data"""
            chunk_idx, (start_row, end_row) = chunk_info
            chunk_start_time = time.time()

            try:
                # Generate output path for this chunk
                chunk_output = Path(self.temp_dir) / f"chunk_{chunk_idx:04d}.jsonl"

                # Transform this chunk
                output_path = transform_parquet_to_jsonl(
                    self.parquet_path_str,
                    str(chunk_output),
                    row_range=(start_row, end_row),
                )

                # Verify output was created
                if not Path(output_path).exists():
                    raise Exception(f"Chunk {chunk_idx} failed to create output file")

                chunk_time = time.time() - chunk_start_time
                return {
                    "chunk_idx": chunk_idx,
                    "start_row": start_row,
                    "end_row": end_row,
                    "output_path": output_path,
                    "processing_time": chunk_time,
                    "success": True,
                    "error": None,
                }

            except Exception as e:
                return {
                    "chunk_idx": chunk_idx,
                    "start_row": start_row,
                    "end_row": end_row,
                    "output_path": None,
                    "processing_time": time.time() - chunk_start_time,
                    "success": False,
                    "error": str(e),
                }

        # Process chunks in parallel
        chunk_inputs = list(enumerate(self.chunks))
        parallel_start_time = time.time()

        self.chunk_results = parallel_map(process_chunk, chunk_inputs)

        self.parallel_processing_time = time.time() - parallel_start_time

        # Analyze results
        successful_chunks = [r for r in self.chunk_results if r["success"]]
        failed_chunks = [r for r in self.chunk_results if not r["success"]]

        print(f"Parallel processing completed in {self.parallel_processing_time:.2f}s")
        print(f"Successful chunks: {len(successful_chunks)}/{len(self.chunks)}")

        if failed_chunks:
            print(f"Failed chunks: {len(failed_chunks)}")
            for failed in failed_chunks:
                print(f"  Chunk {failed['chunk_idx']}: {failed['error']}")
            raise Exception(f"{len(failed_chunks)} chunks failed processing")

        self.chunk_files = [r["output_path"] for r in successful_chunks]
        self.next(self.merge_results)

    @resources(memory=1000, cpu=1)
    @step
    def merge_results(self):
        """Merge processed chunks and create final train/valid split"""
        import json
        import time

        import polars as pl

        print("Merging processed chunks...")
        merge_start_time = time.time()

        # Create output directory
        output_path = Path(self.output_dir_str)
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge all chunk files into temporary combined file
        temp_combined = Path(self.temp_dir) / "combined.jsonl"
        merge_result = merge_jsonl_files(self.chunk_files, str(temp_combined))

        print(
            f"Merged {merge_result['files_merged']} files with {merge_result['total_lines_merged']} total lines"
        )

        # Load combined data and split into train/valid
        df = pl.read_ndjson(temp_combined)
        total_rows = len(df)
        split_point = int(total_rows * 0.95)

        print(
            f"Splitting {total_rows:,} rows: {split_point:,} train, {total_rows - split_point:,} validation"
        )

        # Split data
        train_df = df.slice(0, split_point)
        valid_df = df.slice(split_point)

        # Write final files
        self.train_file = output_path / "train.jsonl"
        self.valid_file = output_path / "valid.jsonl"

        with open(self.train_file, "w", encoding="utf-8") as f:
            for row in train_df.iter_rows(named=True):
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

        with open(self.valid_file, "w", encoding="utf-8") as f:
            for row in valid_df.iter_rows(named=True):
                json.dump(row, f, ensure_ascii=False)
                f.write("\n")

        self.merge_time = time.time() - merge_start_time

        # Calculate statistics
        self.train_samples = len(train_df)
        self.valid_samples = len(valid_df)
        self.total_samples = self.train_samples + self.valid_samples

        print(f"Merge completed in {self.merge_time:.2f}s")
        print("Final files created:")
        print(f"  Train: {self.train_file} ({self.train_samples:,} samples)")
        print(f"  Valid: {self.valid_file} ({self.valid_samples:,} samples)")

        self.next(self.train_model)

    @step
    def train_model(self):
        """Train the model using the prepared data"""
        import time

        print("Starting model training...")
        train_start_time = time.time()

        data_path = self.output_dir_str

        print(f"Training data path: {data_path}")
        print(f"Backend: {self.backend}")
        print(f"Model: {self.model_name}")
        print(f"Training iterations: {self.training_iters}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"LoRA rank: {self.lora_rank}")
        print(f"Max seq length: {self.max_seq_length}")

        self.adapter_dir = train_model(
            data=data_path,
            iters=int(self.training_iters),  # type: ignore
            batch_size=int(self.batch_size),  # type: ignore
            backend=str(self.backend),
            model=str(self.model_name),
            chat_template=str(self.chat_template),
            learning_rate=float(self.learning_rate),  # type: ignore
            lora_rank=int(self.lora_rank),  # type: ignore
            max_seq_length=int(self.max_seq_length),  # type: ignore
        )

        self.training_time = time.time() - train_start_time
        print(f"Model training completed in {self.training_time:.2f}s")

        if str(self.export_format) != "none":
            self.next(self.export_model)
        else:
            self.next(self.end)

    @step
    def export_model(self):
        """Export merged model if export format is specified"""
        print(f"Exporting model as {self.export_format}...")
        export_model(self.adapter_dir, export_format=str(self.export_format))
        self.next(self.end)

    @step
    def end(self):
        """Final validation and cleanup"""

        # Cleanup temporary directory
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

        # Final statistics
        total_time = (
            self.metadata_time
            + self.parallel_processing_time
            + self.merge_time
            + self.training_time
        )

        print("\n" + "=" * 60)
        print("PARALLEL DATA PROCESSING COMPLETED".center(60))
        print("=" * 60)
        print(f"Total rows: {self.total_rows:,} | Chunks: {self.num_chunks}")
        print(
            f"Train: {self.train_samples:,} ({self.train_samples / self.total_samples * 100:.1f}%)"
        )
        print(
            f"Valid: {self.valid_samples:,} ({self.valid_samples / self.total_samples * 100:.1f}%)"
        )
        print("\nTiming:")
        print(
            f"Metadata: {self.metadata_time:.2f}s | Parallel: {self.parallel_processing_time:.2f}s"
        )
        print(f"Merge: {self.merge_time:.2f}s | Training: {self.training_time:.2f}s")
        print(f"Total: {total_time:.2f}s")
        print("\nOutput:")
        print(f"{self.train_file}")
        print(f"{self.valid_file}")
        print("=" * 60)


if __name__ == "__main__":
    ParallelDataFlow()
