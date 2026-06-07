import os
import shutil
import tempfile
from pathlib import Path

import polars as pl
from metaflow import Parameter, parallel_map, resources, step
from metaflow.flowspec import FlowSpec

from data_prep.convert import convert_documents
from data_prep.create_training_data import merge_jsonl_files
from data_prep.generate import generate_dataset
from data_prep.transform import transform_parquet_to_jsonl


class DataPrepFlow(FlowSpec):
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

    data_output = Parameter(
        "data-output",
        help="Directory for final train.jsonl and valid.jsonl output",
        default="",
    )

    chunk_size = Parameter(
        "chunk-size",
        help="Number of rows per chunk for parallel processing",
        default=100 if os.getenv("TEST_MODE") == "true" else 1000,
        type=int,
    )

    num_threads = Parameter(
        "num-threads",
        help="Number of threads per document converter",
        default=4,
        type=int,
    )

    teacher_model = Parameter(
        "teacher-model",
        help="Model identifier for the teacher model",
        default="deepseek-chat",
    )

    num_examples = Parameter(
        "num-examples",
        help="Total QA pairs to generate",
        default=200,
        type=int,
    )

    generation_chunk_size = Parameter(
        "generation-chunk-size",
        help="Words per chunk fed to the teacher model",
        default=2000,
        type=int,
    )

    api_base = Parameter(
        "api-base",
        help="OpenAI-compatible API endpoint URL",
        default="https://api.deepseek.com/v1",
    )

    api_key = Parameter(
        "api-key",
        help="API key (falls back to OPENAI_API_KEY env var)",
        default="",
    )

    @resources(memory=1000, cpu=1)
    @step
    def start(self):
        self._validate_docs_path()
        self.next(self.convert_documents)

    def _validate_docs_path(self):
        self.docs_path_str: str = str(self.docs_path)
        if not self.docs_path_str:
            raise ValueError("--docs-path is required")

        if not Path(self.docs_path_str).is_dir():
            raise ValueError(
                f"--docs-path must be an existing directory: {self.docs_path_str}"
            )

        if self.data_output:
            self.output_dir_str: str = str(self.data_output)
        else:
            self.output_dir_str: str = str(self.script_dir.parent / "LLM" / "data")

        self.chunk_size_int: int = int(self.chunk_size)

        if self.docs_output:
            self.docs_output_str: str = str(self.docs_output)
        else:
            self.docs_output_str: str = str(Path(self.docs_path_str) / "converted")

        print("Starting data preparation pipeline")
        print(f"Documents: {self.docs_path_str}")
        print(f"Output: {self.output_dir_str}")

    @resources(memory=4000, cpu=8)
    @step
    def convert_documents(self):
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
        from openai import OpenAI

        api_key = str(self.api_key) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "--api-key or OPENAI_API_KEY environment variable is required"
            )

        client = OpenAI(api_key=api_key, base_url=str(self.api_base))

        output_dir = str(Path(self.converted_parquet_path).parent / "generated")

        self.parquet_path_str = generate_dataset(
            input_parquet=self.converted_parquet_path,
            output_dir=output_dir,
            client=client,
            model=str(self.teacher_model),
            num_examples=int(self.num_examples),
            chunk_size=int(self.generation_chunk_size),
        )

        print(f"Generated QA pairs: {self.parquet_path_str}")

        self.total_rows = len(pl.read_parquet(self.parquet_path_str))

        self.next(self.process_chunks)

    @resources(memory=1000, cpu=2)
    @step
    def process_chunks(self):
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
            chunk_idx, (start_row, end_row) = chunk_info
            chunk_start_time = time.time()

            try:
                chunk_output = Path(self.temp_dir) / f"chunk_{chunk_idx:04d}.jsonl"

                output_path = transform_parquet_to_jsonl(
                    self.parquet_path_str,
                    str(chunk_output),
                    row_range=(start_row, end_row),
                )

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

        chunk_inputs = list(enumerate(self.chunks))
        parallel_start_time = time.time()

        self.chunk_results = parallel_map(process_chunk, chunk_inputs)

        self.parallel_processing_time = time.time() - parallel_start_time

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
        import json
        import time

        import polars as pl

        print("Merging processed chunks...")
        merge_start_time = time.time()

        output_path = Path(self.output_dir_str)
        output_path.mkdir(parents=True, exist_ok=True)

        temp_combined = Path(self.temp_dir) / "combined.jsonl"
        merge_result = merge_jsonl_files(self.chunk_files, str(temp_combined))

        print(
            f"Merged {merge_result['files_merged']} files with {merge_result['total_lines_merged']} total lines"
        )

        df = pl.read_ndjson(temp_combined)
        total_rows = len(df)
        split_point = int(total_rows * 0.95)

        print(
            f"Splitting {total_rows:,} rows: {split_point:,} train, {total_rows - split_point:,} validation"
        )

        train_df = df.slice(0, split_point)
        valid_df = df.slice(split_point)

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

        self.train_samples = len(train_df)
        self.valid_samples = len(valid_df)
        self.total_samples = self.train_samples + self.valid_samples

        print(f"Merge completed in {self.merge_time:.2f}s")
        print("Final files created:")
        print(f"  Train: {self.train_file} ({self.train_samples:,} samples)")
        print(f"  Valid: {self.valid_file} ({self.valid_samples:,} samples)")

        self.next(self.end)

    @step
    def end(self):
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

        print("\n" + "=" * 60)
        print("DATA PREPARATION COMPLETED".center(60))
        print("=" * 60)
        print(f"Total rows: {self.total_rows:,} | Chunks: {self.num_chunks}")
        print(
            f"Train: {self.train_samples:,} ({self.train_samples / self.total_samples * 100:.1f}%)"
        )
        print(
            f"Valid: {self.valid_samples:,} ({self.valid_samples / self.total_samples * 100:.1f}%)"
        )
        print(f"\nMerge: {self.merge_time:.2f}s")
        print("\nOutput:")
        print(f"{self.train_file}")
        print(f"{self.valid_file}")
        print("=" * 60)


if __name__ == "__main__":
    DataPrepFlow()
