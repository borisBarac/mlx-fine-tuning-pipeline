import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

E2E_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = E2E_DIR.parent
FLOW_FILE = PROJECT_ROOT / "src" / "pipeline.py"
FIXTURE_PDF = E2E_DIR / "LordRings.pdf"


@pytest.fixture
def pipeline_dirs(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shutil.copy2(FIXTURE_PDF, docs_dir / "LordRings.pdf")

    converted_dir = tmp_path / "converted"
    data_dir = tmp_path / "data"
    for d in [converted_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return docs_dir, converted_dir, data_dir


def test_pipeline_e2e(pipeline_dirs, openrouter_client, teacher_model):
    docs_dir, converted_dir, data_dir = pipeline_dirs

    cmd = [
        sys.executable,
        str(FLOW_FILE),
        "run",
        "--docs-path", str(docs_dir),
        "--docs-output", str(converted_dir),
        "--data-output", str(data_dir),
        "--num-threads", "1",
        "--num-examples", "20",
        "--generation-chunk-size", "2000",
        "--training-iters", "2",
        "--batch-size", "1",
        "--model", "unsloth/gemma-3-1b-it",
        "--chat-template", "gemma-3",
        "--learning-rate", "2e-4",
        "--lora-rank", "4",
        "--max-seq-length", "512",
        "--export-format", "none",
        "--teacher-model", teacher_model,
    ]

    env = {
        **os.environ,
        "TEST_MODE": "true",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Pipeline failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    assert (data_dir / "train.jsonl").exists()
    assert (data_dir / "valid.jsonl").exists()

    with open(data_dir / "train.jsonl") as f:
        train_count = sum(1 for line in f if line.strip())
    assert train_count > 0
