from models import send_to_model, get_poem_embedding, get_similarity, sim_to_nat_lang
import datetime
import json

# This is the self-generated-advice version of the file.

class Agent:

    def __init__(self, name):
        self.name = name
        self.poem_list = []
        self.review_list = []
        self.advice = []
        self.known_poems = []
        self.known_poem_embeddings = []

    def export_format(self):
        export = []
        for poem in self.poem_list:
            export.append(
                    {"lines": poem.split("\n"), # TEST WHEN LAB MACHINES UP AGAIN
                     "topic": self.name
                    })

        return export

# BEGIN AGENT CREATION PROCESS HERE

POEM_WRITING_INSTRUCTIONS_PART_1  = ("You are a poet. "
                               "If the user's message is exactly the single "
	                           "word POEM, respond with exactly one original "
	                           "poem and nothing else: no "
	                           "title, no preamble, no explanation, no "
                               "quotation marks. Your aim is to create a poem "
		                       "that will be rated well by your fellow poets. "
                               "Below is a benchmark poem that you and your fellow poets "
	                           "have already seen, and which you all agreed was "
	                           "very good:")

POEM_WRITING_INSTRUCTIONS_PART_2 = ("In general your fellow poets will like poems "
                                    "that are similar to the benchmark, but if they "
                                    "see too many poems like it, they will find them "
                                    "boring and no longer like them. You should aim to "
                                    "create variations on the benchmark poem that are "
                                    "different from it, but not too different.")

POEM_REVIEWING_INSTRUCTIONS = ("You are a reviewer of poems. The user will send "
                              "you a poem and a short comment mentioning how "
                              "similar this poem is to poems that you have already "
                              "seen. You will review the quality of the "
                              "poem by giving it a score out of 10 followed by a \"---\" " 
                              "and a brief explanation of your review. In "
                              "general you will rate poems highly if they are new and "
                              "exciting compared to ones you have seen before. "
                              "You will give medium ratings to poems which are "
                              "similar to ones you have seen before, and low "
                              "ratings to poems that are unfamiliar to you and also "
                              "to poems which are the exact same as poems you "
                              "have seen before. An example review could be: "
                              "**3/10 --- this poem is strange and unfamiliar,  "
                              "with unconventional depictions of mountains**.")

GENERATE_ADVICE = ("You are an advisor to a poet. The user will send you three poems "
                  "and each poem will be followed by a review of the poem by the poet's "
                  "critics. Based on the reviews of the different poems you will "
                  "formulate a one sentence piece of advice about what topics to pursue "
                  "or avoid in the poems. For example if a poem about icebergs is rated "
                  "poorly, but another poem is praised for its haunting atmosphere you "
                  "could write \"Avoid writing about icebergs and aim for a haunting "
                  "atmosphere.\"")

INITIAL_POEM = ("There is a voice, not understood by all,\n"
               "Sent from these desert-caves. It is the roar\n"
               "Of the rent ice-cliff which the sunbeams call,\n"
               "Plunging into the vale--it is the blast\n"
               "Descending on the pines--the torrents pour...")
                            
POEM_WRITING_INSTRUCTIONS = (f"{POEM_WRITING_INSTRUCTIONS_PART_1}\n\n" 
                             f"{INITIAL_POEM}\n\n{POEM_WRITING_INSTRUCTIONS_PART_2}")

# Initialise agents:
agent_agacency_matrix_2 = [[0, 1],
                           [1, 0]]

agent_agacency_matrix_3 = [[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]]

agent_agacency_matrix_4 = [[0, 1, 1, 0],
                           [1, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 1, 0]]

#agents = [0] * len(agent_agacency_matrix)

#for i in range(len(agents)):
#    agents[i] = Agent(f"agent_{i}")


def run_single_agent_test(num_days, agents):

    for agent in agents:
        agent.known_poems.append(INITIAL_POEM)
        agent.known_poem_embeddings.append(get_poem_embedding(poem=INITIAL_POEM))
     
    # Iterate multiple days for the agents:
    
    for day in range(num_days):
        for agent in agents:
            
            # EVERY 3 DAYS GENERATE NEW ADVICE
            if day > 0 and day % 3 == 0:

                advice_prompt = ""

                for poem_num in range(day)[-3:]:
                    assert poem_num + 4 > day
                    advice_prompt += f"POEM: \n {agent.poem_list[poem_num]} \n\n"
                    advice_prompt += f"REVIEW: \n {agent.review_list[poem_num]} \n\n"

                advice = send_to_model(
                            agent=agent.name,
                            prompt_tag="ADVICE",
                            context=GENERATE_ADVICE,
                            prompt = advice_prompt
                        )

                agent.advice.append(advice)

            print(f"Day {day} for {agent.name}: \n\n")
            
            # CONSTRUCT THE POEM GENERATING PROMPT
            context = POEM_WRITING_INSTRUCTIONS + "\n\n"

            if agent.advice:

                context += ("The following are pieces of advice to consider "
                           "when writing your new poem: \n\n")
    
                for advice in agent.advice[-5:]:
                    context += "ADVICE: " + advice + "\n\n"
            
            print("Context: \n", context)

            # GENERATE A NEW POEM
            new_poem = send_to_model(
                           agent=agent.name,
                           prompt_tag="POEM",
                           context=context, 
                           prompt="POEM", 
                       )

            # ANALYSE POEM EMBEDDING
            new_poem_embedding = get_poem_embedding(poem=new_poem)

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

            print("\n")

            # REVIEW THE POEM
            review_prompt = new_poem + "\n\n" + sim_to_nat_lang(similarity)
                
            review = send_to_model(
                agent=agent.name,
                prompt_tag="REVIEW",
                context=POEM_REVIEWING_INSTRUCTIONS, 
                prompt=review_prompt
            )

            # UPDATE AGENT MEMORY
            agent.review_list.append(review) 
            agent.poem_list.append(new_poem)
            agent.known_poems.append(new_poem)
            agent.known_poem_embeddings.append(new_poem_embedding)

            # EVERY 3 DAYS GENERATE NEW ADVICE
            if day > 0 and day % 3 == 0:

                advice_prompt = ""

                for poem_num in range(day)[-3:]:
                    assert poem_num + 4 > day
                    advice_prompt += f"POEM: \n {agent.poem_list[poem_num]} \n\n"
                    advice_prompt += f"REVIEW: \n {agent.review_list[poem_num]} \n\n"

                advice = send_to_model(
                            agent=agent.name,
                            prompt_tag="ADVICE",
                            context=GENERATE_ADVICE,
                            prompt = advice_prompt
                        )

                agent.advice.append(advice)


    x = datetime.datetime.now()
    time_stamp = x.strftime("%b-%d-%H:%M")

    with open(f"{agent.name}_self_advice_{time_stamp}.json", "w") as file:
        json.dump(agents[0].export_format(), file)


NUM_AGENTS = 1

agents = [0] * NUM_AGENTS

for i in range(NUM_AGENTS):
    agents[i] = Agent(f"agent_{i}")

run_single_agent_test(20, agents)

