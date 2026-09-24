import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
from threading import Thread

gen_model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name)


emb_model_name = "Qwen/Qwen3-Embedding-0.6B"
emb_model = SentenceTransformer(emb_model_name)

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
    thread = Thread(target=gen_model.generate, kwargs=dict(**inputs, max_new_tokens=50, streamer=streamer))
    thread.start()

    new_haiku = ""

    # print(text, end="", flush=True)
    for token in streamer:
        print(token, end="", flush=True)
        new_haiku += token

    messages[0]["content"] += "\nHAIKU:\n" + new_haiku

    haiku_embedding = emb_model.encode(new_haiku)
    print("Embedding of HAIKU: " + str(haiku_embedding))
    print("\n")
