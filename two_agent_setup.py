import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

gen_model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)

starting_context = "You are a poet.\nIf the user's message is exactly the single word POEM, respond with exactly one original poem of exactly 12 lines and nothing else: no title, no preamble, no explanation, not quotation marks.\nIf the user's message is the word REVIEW followed by a poem, you are to respond with 1 sentence containing a rating out of 10 for the poem and your reason for the rating.\n"

agent_contexts = [starting_context, starting_context]

for i in range(1):

    for agent_num in range(len(agent_contexts)):
        agent_contexts[agent_num] += f"\nOn day {i + 1} you wrote this: \n"

        messages = [
            {"role": "system", "content": agent_contexts[agent_num]},
            {"role": "user", "content":     "POEM"},
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(target=gen_model.generate, kwargs=dict(**inputs, max_new_tokens=200, streamer=streamer))
        thread.start()
        
        new_poem = ""

        for token in streamer:
#            print(token, end="", flush=True)
            new_poem += token

        agent_contexts[agent_num] += "\nYOUR POEM:\n" + new_poem + "\n"

    for agent_num in range(len(agent_contexts)):

        agent_contexts[agent_num] += f"\nThat same day a fellow poet wrote this: \n"


for context in agent_contexts:
    print("-"*70)
    print(context)



