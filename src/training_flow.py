import time
from pathlib import Path

from metaflow import Parameter, resources, step
from metaflow.flowspec import FlowSpec

from training.trainer import export_model, train_model


class TrainingFlow(FlowSpec):
    data_path = Parameter(
        "data-path",
        help="Directory containing train.jsonl and valid.jsonl",
        default="",
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
        help="Training backend: auto, mlx, or cuda",
        default="auto",
    )

    model_name = Parameter(
        "model",
        help="HuggingFace model identifier",
        default="unsloth/granite-4.0-1b",
    )

    chat_template = Parameter(
        "chat-template",
        help="Chat template name",
        default=None,
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

    @step
    def start(self):
        self.data_path_str: str = str(self.data_path)
        if not self.data_path_str:
            raise ValueError("--data-path is required")

        data_dir = Path(self.data_path_str)
        if not data_dir.is_dir():
            raise ValueError(
                f"--data-path must be an existing directory: {self.data_path_str}"
            )

        train_file = data_dir / "train.jsonl"
        if not train_file.exists():
            raise ValueError(
                f"train.jsonl not found in --data-path: {self.data_path_str}"
            )

        print("Starting training pipeline")
        print(f"Data path: {self.data_path_str}")
        self.next(self.train_model)

    @resources(memory=16000, cpu=4, gpu=1)
    @step
    def train_model(self):
        print("Starting model training...")
        train_start_time = time.time()

        data_path = self.data_path_str

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
            iters=int(self.training_iters),
            batch_size=int(self.batch_size),
            backend=str(self.backend),
            model=str(self.model_name),
            chat_template=str(self.chat_template),
            learning_rate=float(self.learning_rate),
            lora_rank=int(self.lora_rank),
            max_seq_length=int(self.max_seq_length),
        )

        self.training_time = time.time() - train_start_time
        print(f"Model training completed in {self.training_time:.2f}s")

        if str(self.export_format) != "none":
            self.next(self.export_model)
        else:
            self.next(self.end)

    @step
    def export_model(self):
        print(f"Exporting model as {self.export_format}...")
        export_model(self.adapter_dir, export_format=str(self.export_format), backend=str(self.backend))
        self.next(self.end)

    @step
    def end(self):
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED".center(60))
        print("=" * 60)
        print(f"Training time: {self.training_time:.2f}s")
        print(f"Adapter dir: {self.adapter_dir}")
        print("=" * 60)


if __name__ == "__main__":
    TrainingFlow()
