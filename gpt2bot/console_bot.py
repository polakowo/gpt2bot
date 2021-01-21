import random

from .utils import (
    setup_logger,
    load_generator_pipeline,
    clean_text,
    generate_text,
    load_classifier_pipeline,
    classify_responses
)

logger = setup_logger(__name__)


def start_command():
    print("Bot >>>", "Just start texting me. "
                     "If I'm getting annoying, type \"/start\". "
                     "To quit the chat, press Ctrl-C.")
    return []


def run_bot(**kwargs):
    """Runs the console bot.

    kwargs should have three keys:

    * generator_pipeline: Keyword arguments passed to the pipeline (text generation),
    * generator: Keyword arguments passed to the pipeline object (text generation) + seed,
    * classifier_pipeline: Keyword arguments passed to the pipeline (text classification),
    * classifier: Keyword arguments passed to the pipeline object (text classification),
    * chatbot: Keyword arguments for setting up the chatbot.
    """
    generator_pipeline_kwargs = kwargs.get('generator_pipeline', {})
    generator_kwargs = kwargs.get('generator', {})
    classifier_pipeline_kwargs = kwargs.get('classifier_pipeline', {})
    classifier_kwargs = kwargs.get('classifier', {})
    chatbot_kwargs = kwargs.get('chatbot', {})

    # Prepare the pipelines
    generator_pipeline = load_generator_pipeline(**generator_pipeline_kwargs)
    if classifier_pipeline_kwargs['model'] is None:
        classifier_pipeline = None
    else:
        classifier_pipeline = load_classifier_pipeline(**classifier_pipeline_kwargs)

    # Run the chatbot
    logger.info("Running the chatbot...")

    turns = start_command()
    max_turns_history = chatbot_kwargs.get('max_turns_history', 2)
    larger_is_better = classifier_kwargs.get('larger_is_better', True)
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
                    prompt += clean_text(user_message) + generator_pipeline.tokenizer.eos_token
                for bot_message in turn['bot_messages']:
                    prompt += clean_text(bot_message) + generator_pipeline.tokenizer.eos_token

            # Generate bot messages
            bot_messages = generate_text(prompt, generator_pipeline, **generator_kwargs)
            if len(bot_messages) == 1:
                bot_message = bot_messages[0]
            else:
                if classifier_pipeline is None:
                    bot_message = random.choice(bot_messages)
                else:
                    scores = classify_responses(
                        turn['user_messages'][-1],
                        bot_messages,
                        classifier_pipeline,
                        **classifier_kwargs
                    )
                    if larger_is_better:
                        index = max(range(len(bot_messages)), key=scores.__getitem__)
                    else:
                        index = min(range(len(bot_messages)), key=scores.__getitem__)
                    bot_message = bot_messages[index]
            print("Bot >>>", bot_message)
            turn['bot_messages'].append(bot_message)
    except KeyboardInterrupt:
        exit()
    except:
        raise
