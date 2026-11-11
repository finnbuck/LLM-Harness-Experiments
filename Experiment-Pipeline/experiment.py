from Scripts.instructions import self_advice_instructions 
from Scripts.agents import Agent, run_single_agent_test
from Scripts.embedder import create_embedding_plot_3D, create_embedding_plot_2D
import random
import json

with open("4-16_line_poems_with_topics.json", mode="r", encoding="UTF-8") as file:
    dataset_poems = json.load(file)

initial_poem = "\n".join(random.choice(dataset_poems)["lines"])
print(initial_poem)

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

NUM_AGENTS = 1

agents = [0] * NUM_AGENTS

for i in range(NUM_AGENTS):
    agents[i] = Agent(f"agent_{i}")

agent_name_list = [agent.name for agent in agents]

location = run_single_agent_test(60, 
                                 agents, 
                                 self_advice_instructions, 
                                 initial_poem 
                                )

create_embedding_plot_3D(location, abridged_topics, agent_name_list)
create_embedding_plot_2D(location, abridged_topics, agent_name_list)
