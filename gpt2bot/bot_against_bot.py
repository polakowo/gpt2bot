from .utils import *

logger = setup_logger(__name__)


def run_bot(
        generation_pipeline_kwargs={},
        generator_kwargs={},
        prior_rankers_kwargs={},
        cond_rankers_kwargs={},
        chatbot_kwargs={},
        device=-1,
        seed=None,
        debug=False
):
    """Runs the console bot.

    Args:
        generation_pipeline_kwargs: Parameters of the pipeline (text generation).
        generator_kwargs: Parameters of the pipeline object (text generation).
        prior_rankers_kwargs: Parameters of the `prior` rankers.
        cond_rankers_kwargs: Parameters of the `cond` rankers.
        chatbot_kwargs: Parameters of the chatbot.
        device: Device ordinal for CPU/GPU supports.
        seed: Seed for random number generators.
        debug: Whether to enable debugging.
    """

    # Prepare the pipelines
    generation_pipeline = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
    ranker_dict = build_ranker_dict(device=device, **prior_rankers_kwargs, **cond_rankers_kwargs)

    # Run the chatbot
    logger.info("Running the chatbots...")

    max_turns_history = chatbot_kwargs.get('max_turns_history', 2)
    try:
        while True:
            first_message = input("First message: ")
            num_turns = int(input("Number of turns: "))
            print()
            turns = []
            for i in range(num_turns):
                if max_turns_history == 0:
                    turns = []
                # A single turn is a group of user messages and bot responses right after
                turn = {
                    'bot1_messages': [],
                    'bot2_messages': []
                }
                turns.append(turn)
                if i == 0:
                    turn['bot1_messages'].append(first_message)
                    print(f"Bot 1:\t", first_message)
                for j in range(1, 3):
                    if i == 0:
                        continue
                    # Merge turns into a single prompt (don't forget delimiter)
                    prompt = ""
                    from_index = max(len(turns) - max_turns_history - 1, 0) if max_turns_history >= 0 else 0
                    for turn in turns[from_index:]:
                        # Each turn begins with user messages
                        for bot1_message in turn['bot1_messages']:
                            prompt += clean_text(bot1_message) + generation_pipeline.tokenizer.eos_token
                        for bot2_message in turn['bot2_messages']:
                            prompt += clean_text(bot2_message) + generation_pipeline.tokenizer.eos_token

                    # Generate bot messages
                    bot_messages = generate_responses(
                        prompt,
                        generation_pipeline,
                        seed=seed,
                        debug=debug,
                        **generator_kwargs
                    )
                    if len(bot_messages) == 1:
                        bot_message = bot_messages[0]
                    else:
                        bot_message = pick_best_response(
                            prompt,
                            bot_messages,
                            ranker_dict,
                            debug=debug
                        )
                    print(f"Bot {j}:\t", bot_message)
                    turn[f'bot{j}_messages'].append(bot_message)
            print()
    except KeyboardInterrupt:
        exit()
    except:
        raise
