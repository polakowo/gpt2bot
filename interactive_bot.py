#  Copyright (c) polakowo
#  Licensed under the MIT license.

import configparser
import argparse
import logging

from model import download_model_folder, load_model
from decoder import generate_response

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def run_chat(model, tokenizer, config):
    # Parse parameters
    turns_memory = config.getint('chatbot', 'turns_memory')

    logger.info("Running the chatbot...")
    turns = []
    print("Bot >>>", "Just start texting me. If I'm getting annoying, type \"Bye\". To quit the chat type \"Quit\".")
    while True:
        prompt = input("User >>> ")
        if turns_memory == 0:
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
        from_index = max(len(turns)-turns_memory-1, 0) if turns_memory >= 0 else 0
        for turn in turns[from_index:]:
            # Each turn begings with user messages
            for message in turn['user_messages']:
                history += message + tokenizer.eos_token
            for message in turn['bot_messages']:
                history += message + tokenizer.eos_token

        # Generate bot messages
        bot_message = generate_response(model, tokenizer, history, config)
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

    # Download model artifacts
    target_dir = download_model_folder(config)

    # Load model and tokenizer
    model, tokenizer = load_model(target_dir, config)

    # Run chatbot with GPT-2
    run_chat(model, tokenizer, config)
    

if __name__ == '__main__':
    main()
