import os

from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def _detect_backend():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mlx"
    except ImportError:
        pass
    raise RuntimeError(
        "No supported accelerator found. "
        "Requires Apple Silicon (MPS) or NVIDIA GPU (CUDA)."
    )


def _get_training_imports(backend):
    if backend == "cuda":
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        return FastLanguageModel, SFTTrainer, SFTConfig, get_chat_template

    from mlx_tune import (
        FastLanguageModel,
        SFTConfig,
        SFTTrainer,
        get_chat_template,
    )

    return FastLanguageModel, SFTTrainer, SFTConfig, get_chat_template


def train_model(
    data="LLM/data",
    iters=600,
    batch_size=4,
    backend="auto",
    model="unsloth/gemma-3-1b-it",
    chat_template=None,
    learning_rate=2e-4,
    lora_rank=64,
    max_seq_length=2048,
):
    if backend == "auto":
        backend = _detect_backend()

    FastLanguageModel, SFTTrainer, SFTConfig, get_chat_template = _get_training_imports(
        backend
    )

    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=lora_rank,
    )

    if chat_template:
        tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(
        "json", data_files={"train": f"{data}/train.jsonl"}, split="train"
    )

    valid_path = f"{data}/valid.jsonl"
    trainer_kwargs = dict(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )

    if os.path.isfile(valid_path):
        valid_dataset = hf_load_dataset(
            "json", data_files={"valid": valid_path}, split="valid"
        )
        trainer_kwargs["eval_dataset"] = valid_dataset

    trainer = SFTTrainer(
        **trainer_kwargs,
        args=SFTConfig(
            per_device_train_batch_size=batch_size,
            max_steps=iters,
            learning_rate=learning_rate,
            output_dir=f"adapters/{backend}",
            logging_steps=10,
            max_length=max_seq_length,
        ),
    )

    print(f"Training started on {backend} backend...")
    trainer.train()
    print("Training completed.")

    adapter_dir = f"adapters/{backend}"
    model_obj.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapters saved to {adapter_dir}")

    return adapter_dir


def export_model(adapter_dir, export_format="safetensors", backend="auto"):
    if export_format != "safetensors":
        print(
            f"Warning: unsupported export format '{export_format}'. No export performed."
        )
        return

    if backend == "auto":
        backend = _detect_backend()

    FastLanguageModel, _, _, _ = _get_training_imports(backend)

    model_obj, tokenizer = FastLanguageModel.from_pretrained(model_name=adapter_dir)
    model_obj.save_pretrained_merged(adapter_dir, tokenizer)
    print(f"Model exported as safetensors to {adapter_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data", default="LLM/data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "cuda"])
    parser.add_argument("--model", default="unsloth/gemma-3-1b-it")
    parser.add_argument("--chat-template", default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    train_model(
        data=args.data,
        iters=args.iters,
        batch_size=args.batch_size,
        backend=args.backend,
        model=args.model,
        chat_template=args.chat_template,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        max_seq_length=args.max_seq_length,
    )
