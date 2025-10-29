import json
from pathlib import Path

from .transform import create_training_data


# Helper for testing, not used in the pipeline
def createTrainingData(parquet_path=None, output_dir=None):
    # Configuration with defaults
    if parquet_path is None:
        parquet_path = "LLM/parquet_sets/train-00000-of-00001.parquet"
    if output_dir is None:
        output_dir = "LLM/data"

    print(f"Creating training data from: {parquet_path}")
    print(f"Output directory: {output_dir}")

    try:
        # Create training data
        train_path, valid_path = create_training_data(parquet_path, output_dir)

        print("\n✅ Success!")
        print(f"📄 Train file: {train_path}")
        print(f"📄 Valid file: {valid_path}")

        # Show statistics
        import os

        train_size = os.path.getsize(train_path)
        valid_size = os.path.getsize(valid_path)

        with open(train_path, "r") as f:
            train_lines = sum(1 for _ in f)
        with open(valid_path, "r") as f:
            valid_lines = sum(1 for _ in f)

        print("\n📊 Statistics:")
        print(
            f"   Train samples: {train_lines} ({train_lines / (train_lines + valid_lines) * 100:.1f}%)"
        )
        print(
            f"   Valid samples: {valid_lines} ({valid_lines / (train_lines + valid_lines) * 100:.1f}%)"
        )
        print(f"   Total samples: {train_lines + valid_lines}")

        # Return results as dictionary
        return {
            "train_path": train_path,
            "valid_path": valid_path,
            "statistics": {
                "train_samples": train_lines,
                "valid_samples": valid_lines,
                "total_samples": train_lines + valid_lines,
                "train_percentage": train_lines / (train_lines + valid_lines) * 100,
                "valid_percentage": valid_lines / (train_lines + valid_lines) * 100,
                "train_size_bytes": train_size,
                "valid_size_bytes": valid_size,
            },
        }

    except Exception as e:
        raise RuntimeError(f"Failed to create training data: {str(e)}")


def merge_jsonl_files(file_paths, output_filename):
    """
    Merge multiple JSONL files into a single file.

    Args:
        file_paths (list): List of file paths to merge
        output_filename (str): Name of the output file

    Returns:
        dict: Statistics about the merge operation
    """
    valid_files = []
    invalid_files = []
    total_lines = 0

    # Validate files
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            invalid_files.append(file_path)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                # Check if file is valid JSONL by trying to parse first few lines
                line_count = 0
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
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

    # Merge valid files
    if valid_files:
        # Create parent directory if it doesn't exist
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
