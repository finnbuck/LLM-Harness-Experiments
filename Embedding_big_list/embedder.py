# Based strictly on the Sentence Transformers Usage example on Qwen3-Embedding's HF page

from sentence_transformers import SentenceTransformer
import json
import umap
import plotly.graph_objects as go

# Only used for testing
import random

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Default prompts look like this: {'query': 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:', 'document': ''}
# The "prompt" argument allows for a new prompt to be passed to the ecode function.

def make_instruction_prompt(task_description):
    return f'Instruct: {task_description}\nQuery:'

instruction_prompt = make_instruction_prompt("Given a text, return a text on a similar topic.")

# Get the poems

with open("4-16_line_poems_with_topics.json", mode="r", encoding="UTF-8") as poem_file:
    poems = json.load(poem_file)

just_lines = ["\n".join(poem["lines"]) for poem in poems]

embeddings = model.encode(just_lines, prompt=instruction_prompt)

print("Embeddings type", type(embeddings), embeddings.shape) # Result: Embeddings type <class 'numpy.ndarray'>

# Test to see if the order of embeddings gets changed
# The similarity is 0.999 or greater for each one so we can be sure the order was not changed between just_lines and embeddings
# They are not perfectly similar because of padding added to the texts when embeddings are done in batches.
# for i in range(30):
#     test_indice = random.randint(0, 400)
#     sample = just_lines[test_indice]
#     sample_embedding = model.encode(sample, prompt=instruction_prompt)
#     print(model.similarity(sample_embedding, embeddings[test_indice]))

# Applying the UMAP
reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=8,
)

Xr = reducer.fit_transform(embeddings)

fig = go.Figure(go.Scattergl(
    x=Xr[:, 0], y=Xr[:, 1],
    mode="markers",
    # hovertext=[str(k) for k in keys],
    hovertext= ["<br>".join(poem["lines"]) for poem in poems],
    hoverinfo="text",
))

# Now just gotta color by topic

topics = [
    "Love & Courtship",
    "Death & Mourning",
    "Nature & the Seasons",
    "War, Politics & Patriotism",
    "Satire & Social Commentary",
    "Philosophical Reflection",
    "Childhood & Family",
    "Memory, Loss & Nostalgia",
    "Melancholy & the Inner Life",
    "Myth, Legend & the Supernatural"
]

fig.update_traces(marker_size=4)
fig.write_html("big_list_embedded.html")

