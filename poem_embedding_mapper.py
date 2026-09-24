from sentence_transformers import SentenceTransformer
import json
import numpy as np
import umap
import matplotlib.pyplot as plt

emb_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

with open("50-random-6-line-poems.json", mode="r", encoding="UTF-8") as poem_file:
    poems = json.load(poem_file)

embeddings = {}

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
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=8,
)

Xr = reducer.fit_transform(X)

plt.scatter(Xr[:,0], Xr[:,1], s=2)
plt.savefig("umap.png", dpi=150)
