import os

from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def train_model(
    data="LLM/data",
    iters=600,
    batch_size=4,
    backend="mlx",
    model="unsloth/granite-4.0-1b",
    chat_template="llama-3.1",
    learning_rate=2e-4,
    lora_rank=64,
    max_seq_length=2048,
):
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=lora_rank,
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(
        "json", data_files={"train": f"{data}/train.jsonl"}, split="train"
    )

    from trl import SFTTrainer
    from transformers import TrainingArguments

    trainer = SFTTrainer(
        model=model_obj,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            max_steps=iters,
            learning_rate=learning_rate,
            output_dir=f"adapters/{backend}",
            logging_steps=10,
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


def export_model(adapter_dir, export_format="safetensors"):
    from unsloth import FastLanguageModel

    model_obj, tokenizer = FastLanguageModel.from_pretrained(model_name=adapter_dir)
    if export_format == "safetensors":
        model_obj.save_pretrained_merged(
            adapter_dir, tokenizer, save_method="merged_16bit"
        )
        print(f"Model exported as safetensors to {adapter_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data", default="LLM/data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--backend", default="mlx", choices=["mlx", "cuda"])
    parser.add_argument("--model", default="unsloth/granite-4.0-1b")
    parser.add_argument("--chat-template", default="llama-3.1")
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
