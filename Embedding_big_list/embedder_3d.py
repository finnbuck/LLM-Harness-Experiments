# Based strictly on the Sentence Transformers Usage example on Qwen3-Embedding's HF page

from sentence_transformers import SentenceTransformer
import json

# For UMAP
import umap

# For t-SNE
from sklearn.manifold import TSNE 
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px

# Only used for testing
import random

# For unique results filenames
import datetime

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# Default prompts look like this: {'query': 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:', 'document': ''}
# The "prompt" argument allows for a new prompt to be passed to the ecode function.

def make_instruction_prompt(task_description):
    return f'Instruct: {task_description}\nQuery:'

# instruction_prompt = make_instruction_prompt("Given a text, return a text on a similar topic.")

# Tested at 15:41 and 15:43 Oct 13:
# Big difference on some poems, but it does not meaningfully improve its seperation of these categories. That's not that surprising though.

instruction_prompt = make_instruction_prompt("Given a poem return a poem with the same topic.")

# Get the poems

with open("4-16_line_poems_with_topics.json", mode="r", encoding="UTF-8") as poem_file:
    poems = json.load(poem_file)

with open("agent0_poems.json") as file:
    agent0_poems = json.load(file)

with open("agent1_poems.json") as file:
    agent1_poems = json.load(file)

poems += agent0_poems + agent1_poems

for poem in poems:
    print(poem)

just_lines = ["\n".join(poem["lines"]) for poem in poems]

embeddings = model.encode(just_lines, prompt=instruction_prompt)

print("Embeddings type", type(embeddings), embeddings.shape) # Result: Embeddings type <class 'numpy.ndarray'>

# Test to see if the order of embeddings gets changed
# The similarity is 0.999 or greater for each one so we can be sure the order was not changed between just_lines and embeddings
# They are not perfectly similar because of padding added to the texts when embeddings are done in batches.
# The aboce note on padding is based on a Claude response CHECK IT INDEPENDENTLY.
# TEST SUCCESSFUL, can trust ordering.
# for i in range(30):
#     test_indice = random.randint(0, 400)
#     sample = just_lines[test_indice]
#     sample_embedding = model.encode(sample, prompt=instruction_prompt)
#     print(model.similarity(sample_embedding, embeddings[test_indice]))

# Setting up UMAP
reducer = umap.UMAP(
        n_neighbors=100,
        min_dist=0.1,
        n_components=3,
        metric="euclidean",
        random_state=8,
)

# # Setting up TSNE
# reducer = TSNE(n_components = 2, random_state = 8)
# embeddings = StandardScaler().fit_transform(embeddings) # In principle should only be used for t-SNE

Xr = reducer.fit_transform(embeddings)

# fig = go.Figure(go.Scattergl(
#     x=Xr[:, 0], y=Xr[:, 1],
#     mode="markers",
#     # hovertext=[str(k) for k in keys],
#     hovertext= ["<br>".join(poem["lines"]) for poem in poems],
#     hoverinfo="text",
# ))

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
    "Myth, Legend & the Supernatural",
]

agent_list = [
    "Agent_0",
    "Agent_1",
]

fig = go.Figure()

html_format_poems = ["<br>".join(poem["lines"]) for poem in poems]

colors = px.colors.qualitative.Dark24
color_count = 0

for topic in topics:
    topic_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == topic]
    fig.add_trace(go.Scatter3d(
        x=[Xr[:, 0][i] for i in topic_i], y=[Xr[:, 1][i] for i in topic_i], z=[Xr[:, 2][i] for i in topic_i],
        mode="markers",
        marker=dict(color=colors[color_count], size=3),
        # hovertext=[str(k) for k in keys],
        hovertext= [html_format_poems[i] for i in topic_i],
        hoverinfo="text",
        name=topic,
    ))
    color_count += 1

for agent in agent_list:
    agent_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == agent]
    fig.add_trace(go.Scatter3d(
        x=[Xr[:, 0][i] for i in agent_i], y=[Xr[:, 1][i] for i in agent_i], z=[Xr[:, 2][i] for i in agent_i],
        mode="lines+markers",
        marker=dict(color=colors[color_count], size=4),
        line=dict(color=colors[color_count]),
        # hovertext=[str(k) for k in keys],
        hovertext= [html_format_poems[i] for i in agent_i],
        hoverinfo="text",
        name=agent,
    ))
    color_count += 1

x = datetime.datetime.now()
time_stamp = x.strftime("%b-%d-%H:%M")
fig.write_html(f"experiment-3D-{time_stamp}.html")

