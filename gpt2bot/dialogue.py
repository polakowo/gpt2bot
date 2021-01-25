from .utils import *

logger = setup_logger(__name__)


def run(**kwargs):
    """Run a dialogue between two bots."""

    # Extract parameters
    general_params = kwargs.get('general_params', {})
    device = general_params.get('device', -1)
    seed = general_params.get('seed', None)
    debug = general_params.get('debug', False)

    generation_pipeline_kwargs = kwargs.get('generation_pipeline_kwargs', {})
    generation_pipeline_kwargs = {**{
        'model': 'microsoft/DialoGPT-medium'
    }, **generation_pipeline_kwargs}

    generator_kwargs = kwargs.get('generator_kwargs', {})
    generator_kwargs = {**{
        'max_length': 1000,
        'do_sample': True,
        'clean_up_tokenization_spaces': True
    }, **generator_kwargs}

    prior_ranker_weights = kwargs.get('prior_ranker_weights', {})
    cond_ranker_weights = kwargs.get('cond_ranker_weights', {})

    chatbot_params = kwargs.get('chatbot_params', {})
    max_turns_history = chatbot_params.get('max_turns_history', 2)

    # Prepare the pipelines
    generation_pipeline = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
    ranker_dict = build_ranker_dict(device=device, **prior_ranker_weights, **cond_ranker_weights)

    # Run the chatbot
    try:
        while True:
            first_message = input("First message: ")
            num_turns = int(input("Number of turns: "))
            logger.info("Running the dialogue...")
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
                    print(f"Bot 1:", first_message)
                for j in range(1, 3):
                    if i == 0 and j == 1:
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
                    print(f"Bot {j}:", bot_message)
                    turn[f'bot{j}_messages'].append(bot_message)
            print()
    except KeyboardInterrupt:
        exit()
    except:
        raise
