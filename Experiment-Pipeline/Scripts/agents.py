from Scripts.models import send_to_model, get_embedding, get_similarity, sim_to_nat_lang
import datetime
import json

class Agent:

    def __init__(self, name):
        self.name = name
        self.poem_list = []
        self.reviews = {} # Dictionary of Day : [reviewlist pairs]
        self.advice = []
        self.known_poems = []
        self.known_poem_embeddings = []

    def export_format(self):
        export = []
        for poem in self.poem_list:
            export.append(
                    {"lines": poem.split("\n"), 
                     "topic": self.name
                    })

        return export

def run_single_agent_test(num_days, agents, instructions, initial_poem):

    # Load in the instructions:
    
    POEM_WRITING_INSTRUCTIONS_PART_1 = instructions["poem_writing_part_1"]
    POEM_WRITING_INSTRUCTIONS_PART_2 = instructions["poem_writing_part_2"]

    POEM_WRITING_INSTRUCTIONS = (f"{POEM_WRITING_INSTRUCTIONS_PART_1}\n\n"
                                 f"{initial_poem}\n\n"
                                 f"{POEM_WRITING_INSTRUCTIONS_PART_2}")

    POEM_REVIEWING_INSTRUCTIONS = instructions["poem_review"]
    POEM_ADVICE_INSTRUCTIONS = instructions["poem_advice"]

    for agent in agents:

        # All agents start off knowing the initial poem and also 
        # the embedding of the initial poem.
        agent.known_poems.append(initial_poem)
        agent.known_poem_embeddings.append(get_embedding(poem=initial_poem))


    # Iterate multiple days for the agents:

    for day in range(num_days):
        for agent in agents.keys():

            # EVERY 3 DAYS GENERATE NEW ADVICE
            if day > 0 and day % 3 == 0:

                advice_prompt = ""

                for poem_num in range(day)[-3:]:
                    assert poem_num + 4 > day
                    advice_prompt += f"POEM:\n{agent.poem_list[poem_num]}\n\n"

                    for i in range(3):
                        for review in agent.reviews[day - 1 - i]:
                            advice_prompt += f"REVIEW:\n{review}\n\n"

                advice = send_to_model(
                            agent=agent.name,
                            prompt_tag="ADVICE",
                            context=POEM_ADVICE_INSTRUCTIONS,
                            prompt = advice_prompt
                        )

                agent.advice.append(advice)

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

            new_poem_embedding = get_embedding(poem=new_poem)
            agent.reviews[day] = []

            for critic in agents[agent]:

                # ANALYSE POEM EMBEDDING
                similarity = 0

                for old_embedding in critic.known_poem_embeddings:

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
                    agent=critic.name,
                    prompt_tag="REVIEW",
                    context=POEM_REVIEWING_INSTRUCTIONS,
                    prompt=review_prompt
                )

                agent.reviews[day].append(review)

                if not (critic is agent):
                    critic.known_poems.append(new_poem)
                    critic.known_poem_embeddings.append(new_poem_embedding)

            # UPDATE AGENT MEMORY
            agent.poem_list.append(new_poem)
            agent.known_poems.append(new_poem)
            agent.known_poem_embeddings.append(new_poem_embedding)

    x = datetime.datetime.now()
    time_stamp = x.strftime("%b-%d-%H:%M")

    # Remember that paths are relative to experiment.py (when run from there)
    file_location = f"Poems/self_advice_{time_stamp}.json"

    combined_export_list = []
    for agent in agents.keys():
        combined_export_list += (agent.export_format())

    with open(file_location, "w") as file:
        json.dump(combined_export_list, file)

    return file_location

