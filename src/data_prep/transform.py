import json
import time
from pathlib import Path
from typing import Optional

import polars as pl
from metaflow import parallel_map


def transform_parquet_to_jsonl(
    parquet_path: str,
    output_path: Optional[str] = None,
    row_range: Optional[tuple[int, int]] = None,
) -> str:
    """
    Transform a parquet file with 'instruction' and 'output' columns
    to a JSONL file with HuggingFace chat format messages.

    Args:
        parquet_path: Path to input parquet file
        output_path: Optional output path (auto-generated if not provided)

    Returns:
        Path to the created JSONL file

    Raises:
        FileNotFoundError: If input parquet file doesn't exist
        ValueError: If required columns are missing or data is malformed
    """
    input_file = Path(parquet_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet file not found: {parquet_path}")

    try:
        df = pl.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(f"Failed to read parquet file: {str(e)}")

    required_columns = ["instruction", "output"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if row_range is not None:
        start_row, end_row = row_range
        if start_row < 0 or end_row > len(df) or start_row >= end_row:
            raise ValueError(
                f"Invalid row range: {row_range}. Must be within 0-{len(df) - 1} and start < end"
            )
        df = df.slice(start_row, end_row - start_row)

    df_clean = df.filter(
        (
            pl.col("instruction").is_not_null()
            & (pl.col("instruction").str.len_chars() > 0)
        )
        & (pl.col("output").is_not_null() & (pl.col("output").str.len_chars() > 0))
    )

    if output_path is None:
        input_stem = input_file.stem
        output_file = input_file.parent / f"{input_stem}.jsonl"
    else:
        output_file = Path(output_path)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for row in df_clean.select(["instruction", "output"]).iter_rows(named=True):
                messages = {
                    "messages": [
                        {"role": "user", "content": row["instruction"]},
                        {"role": "assistant", "content": row["output"]},
                    ]
                }
                json.dump(messages, f, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        raise ValueError(f"Failed to write JSONL file: {str(e)}")

    return str(output_file)


def merge_jsonl_files(file_paths, output_filename):
    valid_files = []
    invalid_files = []
    total_lines = 0

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            invalid_files.append(file_path)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                line_count = 0
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    line = line.strip()
                    if line:
                        json.loads(line)
                        line_count += 1

                if line_count > 0:
                    valid_files.append(path)
                else:
                    invalid_files.append(file_path)

        except (json.JSONDecodeError, UnicodeDecodeError):
            invalid_files.append(file_path)

    if valid_files:
        output_path = Path(output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_filename, "w", encoding="utf-8") as output_file:
            for file_path in valid_files:
                with open(file_path, "r", encoding="utf-8") as input_file:
                    for line in input_file:
                        line = line.strip()
                        if line:
                            output_file.write(line + "\n")
                            total_lines += 1

    return {
        "output_file": output_filename,
        "valid_files": [str(f) for f in valid_files],
        "invalid_files": invalid_files,
        "total_lines_merged": total_lines,
        "files_merged": len(valid_files),
        "files_skipped": len(invalid_files),
    }


def split_dataset(combined_jsonl_path: str, output_dir: str) -> tuple[str, str]:
    if combined_jsonl_path and Path(combined_jsonl_path).exists():
        df = pl.read_ndjson(combined_jsonl_path)
    else:
        chunk_files_sorted = sorted(Path(output_dir).parent.glob("chunk_*.jsonl"))
        if chunk_files_sorted:
            dfs = [pl.read_ndjson(f) for f in chunk_files_sorted]
            df = pl.concat(dfs)
        else:
            raise FileNotFoundError("No JSONL data found to split")

    total_rows = len(df)

    if total_rows < 300:
        validation_pct = 0.15
    elif total_rows < 1000:
        validation_pct = 0.10
    else:
        validation_pct = 0.05

    split_point = int(total_rows * (1 - validation_pct))

    print(
        f"Splitting {total_rows:,} rows: {split_point:,} train, "
        f"{total_rows - split_point:,} validation ({validation_pct * 100:.0f}%)"
    )

    train_df = df.slice(0, split_point)
    valid_df = df.slice(split_point)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_file = out / "train.jsonl"
    valid_file = out / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_df.iter_rows(named=True):
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for row in valid_df.iter_rows(named=True):
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    print(f"Train: {train_file} ({len(train_df):,} samples)")
    print(f"Valid: {valid_file} ({len(valid_df):,} samples)")

    return str(train_file), str(valid_file)


def process_parquet_in_chunks(
    parquet_path: str,
    chunk_size: int,
    temp_dir: str,
) -> dict:
    import polars as pl

    total_rows = len(pl.read_parquet(parquet_path))

    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    chunks = []
    for i in range(num_chunks):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, total_rows)
        chunks.append((start_row, end_row))

    print(f"Processing {num_chunks} chunks in parallel")
    print(f"Using temporary directory: {temp_dir}")

    def process_chunk(chunk_info):
        chunk_idx, (start_row, end_row) = chunk_info
        chunk_start_time = time.time()

        try:
            chunk_output = Path(temp_dir) / f"chunk_{chunk_idx:04d}.jsonl"

            output_path = transform_parquet_to_jsonl(
                parquet_path,
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

    chunk_inputs = list(enumerate(chunks))
    parallel_start_time = time.time()

    chunk_results = parallel_map(process_chunk, chunk_inputs)

    parallel_processing_time = time.time() - parallel_start_time

    successful_chunks = [r for r in chunk_results if r["success"]]
    failed_chunks = [r for r in chunk_results if not r["success"]]

    print(f"Parallel processing completed in {parallel_processing_time:.2f}s")
    print(f"Successful chunks: {len(successful_chunks)}/{len(chunks)}")

    if failed_chunks:
        print(f"Failed chunks: {len(failed_chunks)}")
        for failed in failed_chunks:
            print(f"  Chunk {failed['chunk_idx']}: {failed['error']}")
        raise Exception(f"{len(failed_chunks)} chunks failed processing")

    chunk_files = [r["output_path"] for r in successful_chunks]

    return {
        "chunk_files": chunk_files,
        "num_chunks": num_chunks,
        "total_rows": total_rows,
        "failed_chunks": failed_chunks,
        "parallel_processing_time": parallel_processing_time,
    }
