from Scripts.instructions import self_advice_instructions, topic_aware_advice_instructions 
from Scripts.agents import Agent, run_single_agent_test
from Scripts.embedder import create_embedding_plot_3D, create_embedding_plot_2D
from Scripts.networks import SELF_LOOP, PAIR, TRIANGLE, SQUARE
import random
import json
import sys

# Picking a random poem as a starting point

with open("4-16_line_poems_with_topics.json", mode="r", encoding="UTF-8") as file:
    dataset_poems = json.load(file)

initial_poem = "\n".join(random.choice(dataset_poems)["lines"])
print(initial_poem)

# Picking which topics we want to appear in the final embedding.

TOPICS = [
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

abridged_topics = TOPICS[:4]

# Setting up the network of agent objects

network = SQUARE

agents = range(len(network))

# The agents are numbered so that they match the numbers in the 
# network notation in the networks.py file
agents = [Agent(f"agent_{i + 1}") for i in agents]

# Numbers in the network dict need to be replaced with 
# their corresponding Agent objects
agents = {agents[k - 1]: [agents[v - 1] for v in l]  for k, l in network.items()}

agent_name_list = [agent.name for agent in agents]

# Run this once the GPU comes back on line and then delete the next two lines
create_embedding_plot_3D("Poems/self_advice_Nov-12-22:40.json", abridged_topics, agent_name_list)
sys.exit()

location = run_single_agent_test(40, 
                                 agents, 
                                 topic_aware_advice_instructions, 
                                 initial_poem 
                                )

create_embedding_plot_3D(location, abridged_topics, agent_name_list)
# create_embedding_plot_2D(location, abridged_topics, agent_name_list)
