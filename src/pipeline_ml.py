import os
import shutil
import tempfile
from pathlib import Path

from metaflow import Parameter, parallel_map, resources, step  # type: ignore
from metaflow.flowspec import FlowSpec

from data_prep.create_training_data import merge_jsonl_files
from data_prep.load import download_parquet_to_cache
from data_prep.transform import transform_parquet_to_jsonl
from mlx.train import train_model
from utils import is_valid_hf_parquet_link


class ParallelDataFlow(FlowSpec):
    # Get the absolute path to the directory containing this script
    script_dir = Path(__file__).parent.resolve()

    parquet_path = Parameter(
        "parquet-path",
        help="Path to input parquet file",
        default=str(
            script_dir.parent / "LLM" / "parquet_sets" / "train-00000-of-00001.parquet"
        ),
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

    @resources(memory=1000, cpu=1)
    @step
    def start(self):
        """Initialize parallel data processing flow"""
        import time

        import polars as pl

        self.output_dir_str: str = str(self.script_dir.parent / "LLM" / "data")

        # Convert parameters to proper types
        self.parquet_path_str: str = str(self.parquet_path)
        self.chunk_size_int: int = int(self.chunk_size)  # type: ignore  # Metaflow handles type conversion

        print("Starting parallel data processing")
        print(f"Input: {self.parquet_path_str}")
        print(f"Output: {self.output_dir_str}")
        print(f"Chunk size: {self.chunk_size_int} rows")

        # Ensure parquet file exists
        if not Path(self.parquet_path_str).exists():
            if not is_valid_hf_parquet_link(self.parquet_path_str):
                raise ValueError(
                    f"The parquet-path parameter must be a valid local path or a Hugging Face parquet link, and the value we got is: `{self.parquet_path_str}`"
                )

            print("Downloading parquet file...")
            self.parquet_path_str = download_parquet_to_cache(self.parquet_path_str)
            print("Downloading parquet file completed.")

        # Get dataset metadata
        start_time = time.time()
        df_meta = pl.scan_parquet(self.parquet_path_str)
        self.total_rows = df_meta.select(pl.len()).collect().item()
        self.metadata_time = time.time() - start_time

        print(
            f"Dataset contains {self.total_rows:,} rows (metadata loaded in {self.metadata_time:.2f}s)"
        )

        # Calculate chunks
        self.num_chunks = (
            self.total_rows + self.chunk_size_int - 1
        ) // self.chunk_size_int
        self.chunks = []

        for i in range(self.num_chunks):
            start_row = i * self.chunk_size_int
            end_row = min((i + 1) * self.chunk_size_int, self.total_rows)
            self.chunks.append((start_row, end_row))

        print(f"Processing {self.num_chunks} chunks in parallel")
        self.next(self.process_chunks)

    @resources(memory=1000, cpu=2)
    @step
    def process_chunks(self):
        """Process data chunks in parallel using parallel_map"""
        import time

        print(f"Processing {len(self.chunks)} chunks in parallel...")

        # Create temporary directory for chunk outputs
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

        # Use the output directory as data path for training
        data_path = self.output_dir_str

        print(f"Training data path: {data_path}")
        print(f"Training iterations: {self.training_iters}")
        print(f"Batch size: {self.batch_size}")

        # Train the model
        train_model(
            data=data_path,
            iters=int(self.training_iters),  # type: ignore
            batch_size=int(self.batch_size),  # type: ignore
        )

        self.training_time = time.time() - train_start_time
        print(f"Model training completed in {self.training_time:.2f}s")

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
