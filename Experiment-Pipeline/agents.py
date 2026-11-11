from models import send_to_model, get_poem_embedding, get_similarity, sim_to_nat_lang
import datetime
import json

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
                    {"lines": poem.split("\n"), 
                     "topic": self.name
                    })

        return export

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

    x = datetime.datetime.now()
    time_stamp = x.strftime("%b-%d-%H:%M")

    with open(f"{agent.name}_self_advice_{time_stamp}.json", "w") as file:
        json.dump(agents[0].export_format(), file)



