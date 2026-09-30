from sentence_transformers import SentenceTransformer
import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

with open("12linepoems.json", mode="r", encoding="UTF-8") as poem_file:
    poems = json.load(poem_file)

embeddings = {}

print(len(poems))

for poem in poems:
    poem_text = "\n".join(poem["lines"])
    sinister_version = "This poem is a sinister message in disguise:\n" + poem_text
    funny_version = "This poem is a funny joke in disguise:\n" + poem_text
    sinister_encoding = emb_model.encode(sinister_version)
    funny_encoding = emb_model.encode(funny_version)
#    print(poem["title"] + "\n")
#    print(poem_text)
#    print("-" * 70 + "\n")

    embeddings[poem["title"] + "_sinister"] = sinister_encoding 
    embeddings[poem["title"] + "_funny"] = funny_encoding 


# Load poems in that have been generated elsewhere:

with open("agent0_poems.json") as file:
    agent0_poems = json.load(file)

with open("agent1_poems.json") as file:
    agent1_poems = json.load(file)

for i in range(len(agent0_poems)):
    encoding = emb_model.encode(agent0_poems[i])
    embeddings[f"agent0_poem{i}"] = encoding


for i in range(len(agent1_poems)):
    encoding = emb_model.encode(agent1_poems[i])
    embeddings[f"agent1_poem{i}"] = encoding


keys = list(embeddings.keys())
print("\n".join(keys))
X = np.stack([embeddings[k] for k in keys])

print(X.shape)

reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=8,
)

Xr = reducer.fit_transform(X)

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=Xr[:,0], y=Xr[:,1], mode="markers", name="Sample_poems"))


fig = go.Figure(go.Scattergl(
    x=Xr[:, 0], y=Xr[:, 1],
    mode="markers",
    hovertext=[str(k) for k in keys],
    hoverinfo="text",
))

fig.update_traces(marker_size=8)
fig.write_html("12_liners_labelled_umap_funny_and_sinister.html")
