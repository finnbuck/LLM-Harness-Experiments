from Scripts.models import get_embedding
from sentence_transformers import SentenceTransformer
import json

# For UMAP
import umap

import plotly.graph_objects as go
import plotly.express as px

# Only used for testing
import random

# For unique results filenames
import datetime

def embed_combined_poems_list(dataset_location):
    # Get the poems
    
    with open("4-16_line_poems_with_topics.json", mode="r", encoding="UTF-8") as poem_file:
        dataset_poems = json.load(poem_file)
        dataset_poems = dataset_poems[:500] # Temporary, delete this after fixing the issue.
    
    with open(dataset_location, mode="r", encoding="UTF-8") as file:
        agent0_poems = json.load(file)
    
    poems = dataset_poems + agent0_poems
    
    just_lines = ["\n".join(poem["lines"]) for poem in poems]

    print(len(just_lines))
    print(just_lines)
    
    embeddings = get_embedding(poem=just_lines) 
    
    print("Embeddings type", type(embeddings), embeddings.shape) 
    # Result: Embeddings type <class 'numpy.ndarray'>
    
    # Test to see if the order of embeddings gets changed
    # The similarity is 0.999 or greater for each one so we can be sure
    # the order was not changed between just_lines and embeddings
    # TEST SUCCESSFUL, can trust ordering.
    # for i in range(30):
    #     test_indice = random.randint(0, 400)
    #     sample = just_lines[test_indice]
    #     sample_embedding = model.encode(sample, prompt=instruction_prompt)
    #     print(model.similarity(sample_embedding, embeddings[test_indice]))
    
    html_format_poems = ["<br>".join(poem["lines"]) for poem in poems]

    return embeddings, html_format_poems, poems

def create_embedding_plot_2D(dataset_location, topics, agent_name_list):
    
    embeddings, html_format_poems, poems = embed_combined_poems_list(dataset_location)
    
    # Setting up UMAP (should do more research into how to pick parameters)
    reducer = umap.UMAP(
            n_neighbors=100,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=8,
    )
    
    Xr = reducer.fit_transform(embeddings)
    
    # Making the embedding plot
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Dark24
    color_count = 0
    
    for topic in topics:
        topic_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == topic]
        fig.add_trace(go.Scatter(
            x=[Xr[:, 0][i] for i in topic_i], 
            y=[Xr[:, 1][i] for i in topic_i],
            mode="markers",
            marker=dict(color=colors[color_count], size=8),
            hovertext= [html_format_poems[i] for i in topic_i],
            hoverinfo="text",
            name=topic,
        ))
        color_count += 1
    
    for agent in agent_name_list:
        agent_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == agent]
        fig.add_trace(go.Scatter(
            x=[Xr[:, 0][i] for i in agent_i], 
            y=[Xr[:, 1][i] for i in agent_i], 
            mode="lines+markers",
            marker=dict(color=colors[color_count], size=10),
            line=dict(color=colors[color_count]),
            hovertext= [html_format_poems[i] for i in agent_i],
            hoverinfo="text",
            name=agent,
        ))
        color_count += 1
    
    x = datetime.datetime.now()
    time_stamp = x.strftime("%b-%d-%H:%M")
    # Remember that paths are relative to experiment.py (when run from ther)
    fig.write_html(f"Embeddings/experiment-2D-{time_stamp}.html")

def create_embedding_plot_3D(dataset_location, topics, agent_name_list):
    
    embeddings, html_format_poems, poems = embed_combined_poems_list(dataset_location)
    
    # Setting up UMAP (should do more research into how to pick parameters)
    reducer = umap.UMAP(
            n_neighbors=100,
            min_dist=0.1,
            n_components=3,
            metric="euclidean",
            random_state=8,
    )
    
    Xr = reducer.fit_transform(embeddings)
    
    # Making the embedding plot
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Dark24
    color_count = 0
    
    for topic in topics:
        topic_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == topic]
        fig.add_trace(go.Scatter3d(
            x=[Xr[:, 0][i] for i in topic_i], 
            y=[Xr[:, 1][i] for i in topic_i],
            z=[Xr[:, 2][i] for i in topic_i],
            mode="markers",
            marker=dict(color=colors[color_count], size=3),
            # hovertext=[str(k) for k in keys],
            hovertext= [html_format_poems[i] for i in topic_i],
            hoverinfo="text",
            name=topic,
        ))
        color_count += 1
    
    for agent in agent_name_list:
        agent_i = [i for i in range(len(embeddings)) if poems[i]["topic"] == agent]
        fig.add_trace(go.Scatter3d(
            x=[Xr[:, 0][i] for i in agent_i], 
            y=[Xr[:, 1][i] for i in agent_i], 
            z=[Xr[:, 2][i] for i in agent_i],
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
    # Remember that paths are relative to experiment.py (when run from ther)
    fig.write_html(f"Embeddings/experiment-3D-{time_stamp}.html")



