import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [
    {"role": "system", "content":   "You are a poet.\n"
                                    "If the user's message is exactly the single word HAIKU," +
                                    "respond with exactly one original haiku — three lines, 5/7/5 syllables — and nothing else: " +
                                    "no preamble, no explanation, no quotation marks.\n" +
                                    "You have written these haikus previously: \n"},
    {"role": "user", "content": "HAIKU"},
]

for i in range(5):

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    print(messages)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs=dict(**inputs, max_new_tokens=50, streamer=streamer))
    thread.start()

    new_haiku = ""

    # print(text, end="", flush=True)
    for token in streamer:
        print(token, end="", flush=True)
        new_haiku += token

    messages[0]["content"] += "\nHAIKU:\n" + new_haiku

    print("\n")