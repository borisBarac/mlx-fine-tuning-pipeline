import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR = Path(__file__).parent.resolve()
ROOT_DIR = BASE_DIR.parent
FLOW_FILE = ROOT_DIR / "src" / "pipeline.py"

SOURCE_DIR = str(BASE_DIR)
CONVERTED_DIR = str(BASE_DIR / "converted")
DATA_DIR = str(BASE_DIR / "data")

BASE_MODEL = "unsloth/gemma-3-1b-it"
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "deepseek/deepseek-v4-flash")
API_BASE = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
NUM_EXAMPLES = 500
GENERATION_CHUNK_SIZE = 2000

LORA_RANK = 16
LORA_ALPHA = 16
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "linear"
LOAD_IN_4BIT = True


def main():
    start_time = time.time()

    api_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
    )
    if not api_key:
        raise ValueError("OPENAI_API_KEY or OPENROUTER_API_KEY not set in .env")

    cmd = [
        sys.executable,
        str(FLOW_FILE),
        "run",
        "--docs-path", SOURCE_DIR,
        "--docs-output", CONVERTED_DIR,
        "--data-output", DATA_DIR,
        "--model", BASE_MODEL,
        "--teacher-model", TEACHER_MODEL,
        "--api-base", API_BASE,
        "--api-key", api_key,
        "--num-examples", str(NUM_EXAMPLES),
        "--generation-chunk-size", str(GENERATION_CHUNK_SIZE),
        "--lora-rank", str(LORA_RANK),
        "--learning-rate", str(LEARNING_RATE),
        "--max-seq-length", str(MAX_SEQ_LENGTH),
        "--batch-size", str(BATCH_SIZE),
        "--num-threads", "4",
        "--chat-template", "gemma-3",
        "--export-format", "none",
    ]

    env = {**os.environ, "PYTHONPATH": str(ROOT_DIR / "src")}

    result = subprocess.run(cmd, cwd=str(ROOT_DIR), env=env)

    elapsed = time.time() - start_time
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print(f"DONE in {elapsed:.1f}s")
        print("=" * 60)
    else:
        print(f"\nPipeline failed (exit {result.returncode}) after {elapsed:.1f}s")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
