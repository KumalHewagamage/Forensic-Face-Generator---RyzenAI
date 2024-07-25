import torch
import logging
import time
from transformers import set_seed
from transformers import LlamaTokenizer
import qlinear
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

set_seed(123)

model_dir = "Model\directory"
ckpt = "pytorch_llama27b_w_bit_4_awq_fa_lm_amd.pt"

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

print(f"Loading from ckpt: {ckpt}")
            
model = torch.load(ckpt)
model = model.to(torch.bfloat16)

for n, m in model.named_modules():
    if isinstance(m, qlinear.QLinearPerGrp):
        print(f"Preparing weights of layer : {n}")
        m.device = "aie"
        m.quantize_weights()

def decode_prompt(model, tokenizer, prompt, input_ids=None, max_new_tokens=30):
    if input_ids is None:
        print(f"prompt: {prompt}")
        start = time.time()
        inputs = tokenizer(prompt, return_tensors="pt")
        end = time.time()
    else:
        start, end = 0, 0

    print("Input Setup - Elapsed Time: " + str(end - start))

    prompt_tokens = 0
    if prompt is None:
        start = time.time()
        generate_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        end = time.time()
        prompt_tokens = input_ids.shape[1]
    else:
        start = time.time()
        generate_ids = model.generate(inputs.input_ids, max_length=50)
        end = time.time()
        prompt_tokens = inputs.input_ids.shape[1]

    num_tokens_out = generate_ids.shape[1]
    new_tokens_generated = num_tokens_out - prompt_tokens
    generate_time = (end - start)
    time_per_token = (generate_time / new_tokens_generated) * 1e3
    print("Generation - Elapsed Time: " + str(generate_time))

    start = time.time()
    response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    end = time.time()

    print(f"response: {response}")
    print("Tokenizer Decode - Elapsed Time: " + str(end - start))

logging.disable(logging.CRITICAL)

def main():
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit", "q"]:
            print("Exiting chat...")
            break
        decode_prompt(model, tokenizer, prompt)

if __name__ == "__main__":
    main()




