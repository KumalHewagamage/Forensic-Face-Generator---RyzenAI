import torch
import requests
from transformers import set_seed, LlamaTokenizer
import qlinear

# Set the seed for reproducibility
set_seed(123)

# Define model and checkpoint paths
model_dir = "model\location"
ckpt = "pytorch_llama27b_w_bit_4_awq_fa_lm_amd.pt"

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_dir)
model = torch.load(ckpt)
model = model.to(torch.bfloat16)

# Quantize model weights
for n, m in model.named_modules():
    if isinstance(m, qlinear.QLinearPerGrp):
        m.device = "aie"
        m.quantize_weights()

# Define a function to generate responses
def decode_prompt(model, tokenizer, prompt, input_ids=None, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt") if input_ids is None else None
    generate_ids = model.generate(inputs.input_ids if inputs else input_ids, max_length=50)
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response

# Main function to handle API interaction
def main():
    while True:
        try:
            # Fetch the prompt from localhost:7000
            response = requests.get("http://localhost:7000")
            prompt = response.json().get("prompt", "")
            if not prompt:
                continue

            # Generate the response using the model
            generated_text = decode_prompt(model, tokenizer, prompt)

            # Send the generated response to localhost:5000
            requests.post("http://localhost:5000", json={"response": generated_text})
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
