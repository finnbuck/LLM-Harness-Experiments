from models import send_to_model, get_poem_embedding, get_similarity, sim_to_nat_lang
import datetime
import json

class Agent:

    def __init__(self, name):
        self.name = name
        self.poem_list = []
        self.known_poems = []
        self.known_poem_embeddings = []
        self.known_poem_summaries = []

    def export_format(self):
        export = []
        for poem in self.poem_list:
            export.append(
                    {"lines": poem.join("\n"), # TEST WHEN LAB MACHINES UP AGAIN
                     "topic": self.name
                    })

        return export

# BEGIN AGENT CREATION PROCESS HERE

POEM_WRITING_INSTRUCTIONS = ("You are a poet. "
                            "If the user's message is exactly the single "
                            "word POEM, respond with exactly one original "
                            "poem of exactly 12 lines and nothing else: no "
                            "title, no preamble, no explanation, no "
                            "quotation marks. Your aim is to create a poem "
                            "that will be rated well by your fellow poets. "
                            "Your fellow poets will only like poems if they "
                            "are sufficiently similar to the set of poems "
                            "that you have already written. They will however "
                            "also not like poems that are too similar to the "
                            "poems you have previously written. Below is a "
                            "list of summaries of poems you have previously "
                            "written. The first 3 words describe the poem, "
                            "the fourth word is a number representing the "
                            "score out of 10 that that poem received and the "
                            "final score is the critic's reason for the score.")

POEM_SUMMARISING_INSTRUCTIONS = ("You are a poem summarising assistant. "
                                "The user will send you a poem along with a "
                                "critic's review of the poem. You will make "
                                "a 5 word terse summary of the review. The "
                                "first 3 words will describe the topic of the "
                                "poem, the 4th word will be the numerical score "
                                "out of 10 given to the poem by the critic. "
                                "The final word will summarise the critic's opinion. "
                                "An example summary could be: **mountain sunset "
                                "melancholy, 5 boring**.")

POEM_REVIEWING_INSTRUCTIONS = ("You are a reviewer of poems. The user will send "
                              "you a poem and a short comment mentioning how "
                              "similar this poem is to poems that you have already "
                              "seen. You will review the quality of the "
                              "poem by giving it a score out of 10 folowed by a -- " 
                              "and a one sentence explanation of your review. In "
                              "general you will rate poems highly if they are new and "
                              "exciting compared to ones you have seen before. "
                              "You will give medium ratings to poems which are "
                              "similar to ones you have seen before, and low "
                              "ratings to poems that are unfamilar to you and also "
                              "to poems which are the exact same as poems you "
                              "have seen before. An example review could be: "
                              "**3/10 -- this poem is strange and unfamiliar. ")

INITIAL_POEM = ("There is a voice, not understood by all,\n"
               "Sent from these desert-caves. It is the roar\n"
               "Of the rent ice-cliff which the sunbeams call,\n"
               "Plunging into the vale--it is the blast\n"
               "Descending on the pines--the torrents pour...")
                            
# Initialise agents:

agent_agacency_matrix_TEST = [[0, 1],
                            [1, 0]]

agent_agacency_matrix = [[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]]

#agents = [0] * len(agent_agacency_matrix)

#for i in range(len(agents)):
#    agents[i] = Agent(f"agent_{i}")

agents = [0]

# OLD 1-Agent test version below

def run_single_agent_test(num_days):

    for i in range(1):
        agents[i] = Agent("agent_0")
        agents[i].known_poems.append(INITIAL_POEM)
        agents[i].known_poem_embeddings.append(get_poem_embedding(poem=INITIAL_POEM))
     
        prompt = agents[i].known_poems[-1] + "\n\n" + "This poem is new and exciting."
     
        # For testing with one agent the agent reviews their own poems.
        review = send_to_model(
            agent=agents[i].name,
            prompt_tag="REVIEW",
            context=POEM_REVIEWING_INSTRUCTIONS, 
            prompt=prompt, 
        )
    
        prompt = agents[i].known_poems[-1] + "\n\n" + review
    
        agents[i].known_poem_summaries.append(send_to_model(
            agent=agents[i].name,
            prompt_tag="SUMMARISE",
            context=POEM_SUMMARISING_INSTRUCTIONS, 
            prompt=prompt, 
        ))
    
    # Iterate multiple days for the agents:
    
    for day in range(num_days):
        for agent in agents:
            print(f"Day {day} for {agent.name}: \n\n")

            context = POEM_WRITING_INSTRUCTIONS + "\n\n"
    
            for summary in agent.known_poem_summaries:
                context += summary + "\n"
    
            new_poem = send_to_model(
                           agent=agent.name,
                           prompt_tag="POEM",
                           context=context, 
                           prompt="POEM", 
                       )

            new_poem_embedding = get_poem_embedding(poem=new_poem)

            print("Context: \n", context)

            # The loop below finds the maximum similarity of the 
            # new poem to the set of existing poems
            similarity = 0
            for old_embedding in agent.known_poem_embeddings:

                candidate_sim = get_similarity(
                                    new_poem_embedding, 
                                    old_embedding
                                )

                if candidate_sim > similarity:
                    similarity = candidate_sim
                    print("New similarity: ", similarity)

            prompt = new_poem + "\n\n" + sim_to_nat_lang(similarity)
                
            # For testing with one agent the agent reviews their own poems.
            review = send_to_model(
                agent=agent.name,
                prompt_tag="REVIEW",
                context=POEM_REVIEWING_INSTRUCTIONS, 
                prompt=prompt, 
            )

            prompt = new_poem + "\n\n" + review
    
            agents[i].known_poem_summaries.append(send_to_model(
                 agent=agent.name,
                 prompt_tag="SUMMARISE",
                 context=POEM_SUMMARISING_INSTRUCTIONS, 
                 prompt=prompt, 
            ))
            
            agent.poem_list.append(new_poem)
            agent.known_poems.append(new_poem)
            agent.known_poem_embeddings.append(new_poem_embedding)

    x = datetime.datetime.now()
    time_stamp = x.strftime("%b-%d-%H:%M")

    with open(f"{agent.name}_poems_{time_stamp}.json", "w") as file:
        json.dump(agents[0].export_format(), file)

        # The basic structure should be:

        # 1. Write a poem per agent.
        # 2. Review and summarise the poems.
        # 3. Add the right poems and summaries to the right agents (writers and reviewers). 

run_single_agent_test(20)

