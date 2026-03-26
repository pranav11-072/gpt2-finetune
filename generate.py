from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

MODEL_DIR = "./gpt2-finetuned"

# ── Load fine-tuned model ────────────────────────────────────────────────────
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model     = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_text(prompt: str, max_new_tokens: int = 200) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ── Try it out ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    print("\n--- Generated Text ---")
    print(generate_text(prompt))
