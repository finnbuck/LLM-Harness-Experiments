import os
os.environ["HF_HUB_OFFLINE"] = "1"

# modules needed for the huggingface models
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Generative model setup copied from https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

gen_model_name = "Qwen/Qwen2.5-3B-Instruct"

gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_name,
    dtype=torch.float16, # Decided to set manually for more transparancy.
).to("cuda") # Likewise decided to explicitly move process to GPU. 
             # It will fail if no GPU available.

tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

# Embedding model setup from https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# The following function is a reusable version of the method shown in
# https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# Use the keywords "context" and "prompt" to avoid confusion.

def send_to_model(**kwargs):

    for k, v in kwargs.items():
        if k == "context":
            context = v
        elif k == "prompt":
            prompt = v 
        elif k == "agent":
            agent = v
        else:
            print("send_to_model does not regognise the kwarg ", k)
    
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    # Changed to be explicitly to cuda

    generated_ids = gen_model.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"### {prompt} ---> {agent}:\n")
    print(response + "\n")

    return(response)

# TEST
# send_to_model(
#     agent = "agent_1",
#     context = "You are a helpful assistant.",
#     prompt = "Write a short poem."
# )

# Current superfluous but keeping for later use.
def make_instruction_prompt(task_description):
    return f'Instruct: {task_description}\nQuery:'

# The method below is based on the functionality demonstrated in
# https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

def get_poem_embedding(**kwargs):
    for k, v in kwargs.items():
        if k == "poem":
            poem = v
        else:
            print("embed_poem does not regognise the kwarg ", k)
    
    embedding_instruction_prompt = make_instruction_prompt(
        "Given a poem return a poem with the same topic."
    )

    embedding = embedding_model.encode(poem, prompt=embedding_instruction_prompt)

    return(embedding)

def similarity_to_NL(similarity):
    similarity = float(similarity)

    if similarity > 0.99:
        return "the same as"
    elif similarity > 0.95:
        return "very similar to"
    elif similarity > 0.9:
        return "similar to"
    elif similarity > 0.75:
        return "new and exciting compared to"
    elif similarity > 0.5:
        return "strange and different compared to"
    else:
        return "unpleasant and unfamiliar compared to"

# poem_1 = send_to_model(
#     agent = "agent_1",
#     context = "You are a helpful assistant.",
#     prompt = "Write a short poem."
# )

# poem_2 = send_to_model(
#     agent = "agent_1",
#     context = "You are a helpful assistant.",
#     prompt = "Write a short poem."
# )

# embedding_1 = get_poem_embedding(poem = poem_1)
# embedding_2 = get_poem_embedding(poem = poem_2)

# print("Embedding_1: ", embedding_1)
# print("Embedding_1 type: ", type(embedding_1))

# print("Similarity of poem_1 to poem_1", 
#     embedding_model.similarity(embedding_1, embedding_1)
# )

# sim_between = embedding_model.similarity(embedding_1, embedding_2)

# print("Similarity of poem_1 to poem_2", sim_between)

# print("Converted to number: ", similarity_to_NL(sim_between))
