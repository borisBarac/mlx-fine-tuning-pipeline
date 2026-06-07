import sys
from pathlib import Path

_src_dir = str(Path(__file__).parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from training.trainer import train_model  # noqa: E402

E2E_DIR = Path(__file__).parent.resolve()
DATA_DIR = E2E_DIR / "mini_train"

MODEL = "unsloth/gemma-3-1b-it"


def test_training_pipeline_e2e(tmp_path: Path):
    adapter_dir = train_model(
        data=str(DATA_DIR),
        iters=2,
        batch_size=1,
        model=MODEL,
        chat_template="gemma-3",
        learning_rate=2e-4,
        lora_rank=4,
        max_seq_length=512,
    )

    assert Path(adapter_dir).is_dir()
    adapter_files = list(Path(adapter_dir).rglob("*.safetensors"))
    assert len(adapter_files) > 0, f"No adapter files found in {adapter_dir}"
