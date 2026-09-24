from sentence_transformers import SentenceTransformer
import json
import numpy as np
import umap
import matplotlib.pyplot as plt
import plotly.express as px

emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

with open("12linepoems.json", mode="r", encoding="UTF-8") as poem_file:
    poems = json.load(poem_file)

embeddings = {}

print(len(poems))

for poem in poems:
    poem_text = "\n".join(poem["lines"])
    encoding = emb_model.encode(poem_text)
#    print(poem["title"] + "\n")
#    print(poem_text)
#    print("-" * 70 + "\n")
    embeddings[poem["title"]] = encoding 

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

fig = px.scatter(
    x=Xr[:, 0], y=Xr[:, 1],
    hover_name=[str(k) for k in keys],
)
fig.update_traces(marker_size=8)
fig.write_html("12_liners_labelled_umap.html")
