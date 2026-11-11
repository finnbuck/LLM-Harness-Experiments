from instructions import self_advice_instructions 
from agents import Agent, run_single_agent_test

NUM_AGENTS = 1

agents = [0] * NUM_AGENTS

for i in range(NUM_AGENTS):
    agents[i] = Agent(f"agent_{i}")

location = run_single_agent_test(5, 
                                 agents, 
                                 self_advice_instructions, 
                                 "Mary had a little lamb"
                                )

print(location)


