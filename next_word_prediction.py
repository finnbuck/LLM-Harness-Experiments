import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)



text = "The brain is a"
response = ""

print(text, end="")

for i in range(0, 10):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]          # scores for the next token, given the last position
    probs = torch.softmax(logits, dim=-1)      # turn scores into a probability distribution
    next_token_id = torch.argmax(probs)        # pick the single most likely token
    next_token = tokenizer.decode(next_token_id)
    text += next_token
    print(next_token, end="")

print("\n")

