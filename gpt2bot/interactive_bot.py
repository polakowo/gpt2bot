#  Copyright (c) polakowo
#  Licensed under the MIT license.

import configparser
import argparse
import logging
import random

from model import load_model
from decoder import generate_response

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def run_chat(model, tokenizer, config, mmi_model=None, mmi_tokenizer=None):
    # Parse parameters
    num_samples = config.getint('decoder', 'num_samples')
    max_turns_history = config.getint('decoder', 'max_turns_history')

    logger.info("Running the chatbot...")
    turns = []
    print("Bot >>>", "Just start texting me. If I'm getting annoying, type \"Bye\". To quit the chat type \"Quit\".")
    while True:
        prompt = input("User >>> ")
        if max_turns_history == 0:
            # If you still get different responses then set seed
            turns = []
        if prompt.lower() == 'bye':
            print("Bot >>>", "Bye")
            turns = []
            continue
        if prompt.lower() == 'quit':
            break
        # A single turn is a group of user messages and bot responses right after
        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        turns.append(turn)
        turn['user_messages'].append(prompt)
        # Merge turns into a single history (don't forget EOS token)
        history = ""
        from_index = max(len(turns)-max_turns_history-1, 0) if max_turns_history >= 0 else 0
        for turn in turns[from_index:]:
            # Each turn begings with user messages
            for message in turn['user_messages']:
                history += message + tokenizer.eos_token
            for message in turn['bot_messages']:
                history += message + tokenizer.eos_token

        # Generate bot messages
        bot_messages = generate_response(
            model, 
            tokenizer, 
            history, 
            config, 
            mmi_model=mmi_model, 
            mmi_tokenizer=mmi_tokenizer
        )
        if num_samples == 1:
            bot_message = bot_messages[0]
        else:
            # TODO: Select a message that is the most appropriate given the context
            # This way you can avoid loops
            bot_message = random.choice(bot_messages)
        print("Bot >>>", bot_message)
        turn['bot_messages'].append(bot_message)

def main():
    # Script arguments can include path of the config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default="chatbot.cfg")
    args = arg_parser.parse_args()

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    with open(args.config) as f:
        config.read_file(f)

    # Load main model
    model_folder = config.get('model', 'model_folder')
    model, tokenizer = load_model(model_folder, config)

    # Download and load reverse model
    use_mmi = config.getboolean('model', 'use_mmi')
    if use_mmi:
        mmi_model_folder = config.get('model', 'mmi_model_folder')
        mmi_model, mmi_tokenizer = load_model(mmi_model_folder, config)
    else:
        mmi_model = None
        mmi_tokenizer = None
    
    # Run chatbot with GPT-2
    run_chat(model, tokenizer, config, mmi_model=mmi_model, mmi_tokenizer=mmi_tokenizer)

if __name__ == '__main__':
    main()
