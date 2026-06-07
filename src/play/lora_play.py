from mlx_lm import generate, load  # type: ignore

BASE_MODEL = "unsloth/gemma-3-1b-it"
ADAPTER_PATH = "adapters/lora"


def query_model(prompt: str, max_tokens: int = 50) -> str:
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=formatted, max_tokens=max_tokens)


if __name__ == "__main__":
    try:
        test_prompt = "hi! how are you?"
        print("Sending test prompt to model:", test_prompt)
        response = query_model(test_prompt)
        print("Model response:", response)
    except Exception as e:
        print("Error during model query:", str(e))
