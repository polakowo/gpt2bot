import random

from .utils import setup_logger, load_pipeline, clean_text, generate_text

logger = setup_logger(__name__)


def start_command():
    print("Bot >>>", "Just start texting me. "
                     "If I'm getting annoying, type \"/start\". "
                     "To quit the chat, press Ctrl-C.")
    return []


def run_bot(**kwargs):
    """Runs the console bot.

    kwargs should have three keys:

    * pipeline: Keyword arguments passed when calling transformers.pipeline,
    * generator: Keyword arguments passed when calling the pipeline object + seed,
    * chatbot: Keyword arguments for setting up the chatbot."""
    pipeline_kwargs = kwargs.get('pipeline', {})
    generator_kwargs = kwargs.get('generator', {})
    chatbot_kwargs = kwargs.get('chatbot', {})

    # Prepare the pipeline
    pipeline = load_pipeline(**pipeline_kwargs)

    # Run the chatbot
    logger.info("Running the chatbot...")

    max_turns_history = chatbot_kwargs.get('max_turns_history', 2)
    message_selector = chatbot_kwargs.get('message_selector', random.choice)
    turns = start_command()
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
                    prompt += clean_text(user_message) + pipeline.tokenizer.eos_token
                for bot_message in turn['bot_messages']:
                    prompt += clean_text(bot_message) + pipeline.tokenizer.eos_token

            # Generate bot messages
            bot_messages = generate_text(prompt, pipeline, **generator_kwargs)
            if len(bot_messages) == 1:
                bot_message = bot_messages[0]
            else:
                bot_message = message_selector(bot_messages)
            print("Bot >>>", bot_message)
            turn['bot_messages'].append(bot_message)
    except KeyboardInterrupt:
        exit()
    except:
        raise
