from Scripts.instructions import self_advice_instructions 
from Scripts.agents import Agent, run_single_agent_test
from Scripts.embedder import create_embedding_plot_3D, create_embedding_plot_2D

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

abridged_topics = TOPICS[:3]

NUM_AGENTS = 1

agents = [0] * NUM_AGENTS

for i in range(NUM_AGENTS):
    agents[i] = Agent(f"agent_{i}")

agent_name_list = [agent.name for agent in agents]

location = run_single_agent_test(5, 
                                 agents, 
                                 self_advice_instructions, 
                                 "Mary had a little lamb"
                                )

create_embedding_plot_3D(location, abridged_topics, agent_name_list)
create_embedding_plot_2D(location, abridged_topics, agent_name_list)
