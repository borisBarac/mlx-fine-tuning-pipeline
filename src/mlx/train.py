import os
import types

from huggingface_hub import login
from mlx_lm.lora import CONFIG_DEFAULTS
from mlx_lm.lora import train_model as lora_train_model
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)


def train_model(data="LLM/data", iters=600, batch_size=4):
    """
    Train the model using LoRA fine-tuning.

    Args:
        data (str): Path to directory containing JSONL files
        model_name (str): Model identifier to load
        iters (int): Number of training iterations
    """

    # Create args object with training parameters
    args = types.SimpleNamespace()
    args.model = "mlx-community/granite-4.0-1b-base-4bit"
    args.data = data
    args.train = True
    args.iters = iters
    args.batch_size = batch_size

    # Set defaults for other required parameters
    for k, v in CONFIG_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Load the pre-trained model
    model, tokenizer, *_ = load(args.model)

    # Set chat template if not present
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"

    # Load local datasets
    train_set, valid_set, _ = load_dataset(args, tokenizer)  # type: ignore

    print("Model and tokenizer loaded successfully.")
    print(f"Training set size: {len(train_set)}")
    print(f"Validation set size: {len(valid_set)}")
    print(f"Total iterations: {args.iters}")
    print("Training started...")

    # Start training
    lora_train_model(args, model, train_set, valid_set)

    print("Training completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--data", default="LLM/data", help="Path to directory containing JSONL files"
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )

    parser.add_argument(
        "--iters", type=int, default=600, help="Number of training iterations"
    )
    args = parser.parse_args()

    train_model(args.data, args.iters, args.batch_size)
