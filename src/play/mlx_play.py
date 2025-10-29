from mlx_lm import generate, load  # type: ignore


def query_model(prompt: str, model_path: str = "adapters", max_tokens: int = 50) -> str:
    model, tokenizer, *_ = load(
        path_or_hf_repo="mlx-community/granite-4.0-1b-base-4bit",
        adapter_path="adapters",  # path to trained adapter
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return response


if __name__ == "__main__":
    try:
        test_prompt = "who won in 2024?"
        print("Sending test prompt to model:", test_prompt)
        response = query_model(test_prompt)
        print("Model response:", response)
    except Exception as e:
        print("Error during model query:", str(e))
