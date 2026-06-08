from unsloth import FastLanguageModel
from peft import PeftModel

BASE_MODEL = "unsloth/gemma-3-1b-it"
ADAPTER_PATH = "adapters/lora"
max_seq_length = 2048
dtype = None
load_in_4bit = True


def query_model(prompt: str, max_tokens: int = 50) -> str:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    try:
        test_prompt = "hi! how are you?"
        print("Sending test prompt to model:", test_prompt)
        response = query_model(test_prompt)
        print("Model response:", response)
    except Exception as e:
        print("Error during model query:", str(e))
