from .utils import *

logger = setup_logger(__name__)


def start_command():
    print("Bot >>>", "Just start texting me. "
                     "If I'm getting annoying, type \"/start\". "
                     "To quit the chat, press Ctrl-C.")
    return []


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
    logger.info("Running the chatbot...")

    turns = start_command()
    max_turns_history = chatbot_kwargs.get('max_turns_history', 2)
    try:
        while True:
            prompt = input("User >>> ")
            if max_turns_history == 0:
                turns = []
            if prompt.lower() == '/start':
                turns = start_command()
                continue
            # A single turn is a group of user messages and bot responses right after
            turn = {
                'user_messages': [],
                'bot_messages': []
            }
            turns.append(turn)
            turn['user_messages'].append(prompt)
            # Merge turns into a single prompt (don't forget delimiter)
            prompt = ""
            from_index = max(len(turns) - max_turns_history - 1, 0) if max_turns_history >= 0 else 0
            for turn in turns[from_index:]:
                # Each turn begins with user messages
                for user_message in turn['user_messages']:
                    prompt += clean_text(user_message) + generation_pipeline.tokenizer.eos_token
                for bot_message in turn['bot_messages']:
                    prompt += clean_text(bot_message) + generation_pipeline.tokenizer.eos_token

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
            print("Bot >>>", bot_message)
            turn['bot_messages'].append(bot_message)
    except KeyboardInterrupt:
        exit()
    except:
        raise
