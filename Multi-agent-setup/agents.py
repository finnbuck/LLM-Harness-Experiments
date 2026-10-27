from models import send_to_model, get_poem_embedding, sim_score_to_nat_lang

class Agent:

    def __init__(self):
        self.poem_list = []
        self.known_poems = []
        self.known_poem_embeddings = []
        self.prev_poem_summaries = []

# BEGIN AGENT CREATION PROCESS HERE

agents = [""]

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
                              "poem by giving it a score out of 10 folowed by a -- and a one "
                              "sentence explanation of your review. In general "
                              "you will rate poems highly if they are new and "
                              "exciting compared to ones you have seen before. "
                              "You will give medium ratings to poems which are "
                              "similar to ones you have seen before, and low "
                              "ratings to poems that are unfamilar to you and also "
                              "to poems which are the exact same as poems you "
                              "have seen before. An example review could be: "
                              "**3/10 -- this poem is strange and unfamiliar. ")

INITIAL_POEM = ["There is a voice, not understood by all,",
               "Sent from these desert-caves. It is the roar",
               "Of the rent ice-cliff which the sunbeams call,",
               "Plunging into the vale--it is the blast",
               "Descending on the pines--the torrents pour..."]
                            
# Initialise agents:

for i in range(1):
    agents[i] = Agent()
    agents[i].known_poems.append("\n".join(INITIAL_POEM))
 
    prompt = agents[i].known_poems[-1] + "\n\n" + "This poem is new and exciting."
 
    # For testing with one agent the agent reviews their own poems.
    review = send_to_model(
        agent=f"agent_{i}",
        context=POEM_REVIEWING_INSTRUCTIONS, 
        prompt=prompt, 
        prompt_tag="REVIEW")

    prompt = agents[i].known_poems[-1] + "\n\n" + review

    agents[i].prev_poem_summaries.append(send_to_model(
        agent=f"agent_{i}",
        context=POEM_SUMMARISING_INSTRUCTIONS, 
        prompt=prompt, 
        prompt_tag="SUMMARISE"
    ))

    context = POEM_WRITING_INSTRUCTIONS + "\n\n" + agents[i].prev_poem_summaries[0]

    agents[i].poem_list.append(send_to_model(
        agent=f"agent_{i}",
        context=context, 
        prompt="POEM", 
        prompt_tag="POEM"
    ))

print(agents)