import shutil
import time
from pathlib import Path

from metaflow import Parameter, resources, step

from data_prep_flow import DataPrepFlow
from training.trainer import TrainingConfig, export_model, train_model


class ParallelDataFlow(DataPrepFlow):
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
        default="unsloth/gemma-3-1b-it",
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

    @resources(memory=1000, cpu=1)
    @step
    def merge_results(self):
        self._do_merge()
        self.next(self.train_model)

    @resources(memory=16000, cpu=4, gpu=1)
    @step
    def train_model(self):
        print("Starting model training...")
        train_start_time = time.time()

        config = TrainingConfig(
            data=self.output_dir_str,
            iters=int(self.training_iters),
            batch_size=int(self.batch_size),
            backend=str(self.backend),
            model=str(self.model_name),
            chat_template=str(self.chat_template) if self.chat_template else None,
            learning_rate=float(self.learning_rate),
            lora_rank=int(self.lora_rank),
            max_seq_length=int(self.max_seq_length),
        )

        print(f"Training config: {config}")

        self.adapter_dir = train_model(
            data=config.data,
            iters=config.iters,
            batch_size=config.batch_size,
            backend=config.backend,
            model=config.model,
            chat_template=config.chat_template,
            learning_rate=config.learning_rate,
            lora_rank=config.lora_rank,
            max_seq_length=config.max_seq_length,
        )

        self.training_time = time.time() - train_start_time
        print(f"Model training completed in {self.training_time:.2f}s")

        self.next(self.export_model_step)

    @step
    def export_model_step(self):
        if str(self.export_format) != "none":
            print(f"Exporting model as {self.export_format}...")
            export_model(
                self.adapter_dir,
                export_format=str(self.export_format),
                backend=str(self.backend),
            )
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

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED".center(60))
        print("=" * 60)
        print(f"Training time: {self.training_time:.2f}s")
        print(f"Adapter dir: {self.adapter_dir}")
        print("=" * 60)


if __name__ == "__main__":
    ParallelDataFlow()
