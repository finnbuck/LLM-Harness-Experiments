import os
import torch
import json
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

gen_model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, torch_dtype=torch.float16).to("cuda")

starting_context = "You are a poet.\nIf the user's message is exactly the single word POEM, respond with exactly one original poem of exactly 12 lines and nothing else: no title, no preamble, no explanation, not quotation marks.\nIf the user's message is the word REVIEW followed by a poem, you are to respond with an extremely short text containing a rating out of 10 for the poem and your reason for the rating.\n"

agent_contexts = [starting_context, starting_context]

agents = []

for num in range(2):
    agents.append({"context": starting_context, "poem_list": []})

for i in range(3):

    for agent in agents:

        agent["context"] += f"\nOn day {i + 1} you wrote this: \n"

        messages = [
            {"role": "system", "content": agent["context"]},
            {"role": "user", "content": "POEM"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(target=gen_model.generate, kwargs=dict(**inputs, max_new_tokens=200, streamer=streamer))
        thread.start()
        
        new_poem = ""

        for token in streamer:
            print(token, end="", flush=True)
            new_poem += token

        agent["context"] += "\nYOUR POEM:\n" + new_poem + "\n"
        agent["poem_list"].append(new_poem)

    for j in range(2):

        messages = [
            {"role": "system", "content": agents[j]["context"]},
            {"role": "user", "content": "REVIEW\n" + agents[(j + 1) % 2]["poem_list"][i]}
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(target=gen_model.generate, kwargs=dict(**inputs, max_new_tokens=200, streamer=streamer))
        thread.start()
        
        new_review = ""

        for token in streamer:
            print(token, end="", flush=True)
            new_review += token

        agents[(j + 1) % 2]["context"] += "\nA fellow poet reviewed your poem as follows: \n" + new_review + "\n"

for i in range(len(agents)):
    with open(f"agent{i}_history.txt", "w", encoding="utf-8") as file:
        file.write(agents[i]["context"]) 

    with open(f"agent{i}_poems.json", "w") as file:
        json.dump(agents[i]["poem_list"], file)

# for agent in agents:
#     print("-"*70)
#     print(agent["context"])
print(len(agents[0]["poem_list"]), len(agents[1]["poem_list"]))


