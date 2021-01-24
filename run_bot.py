import argparse

from gpt2bot.telegram_bot import run as run_telegram_bot
from gpt2bot.console_bot import run as run_console_bot
from gpt2bot.dialogue import run as run_dialogue
from gpt2bot.utils import parse_config

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--type',
        type=str,
        default='telegram',
        help="Type of the conversation to run: telegram, console or dialogue"
    )
    arg_parser.add_argument(
        '--config',
        type=str,
        default='configs/medium-cpu.cfg',
        help="Path to the config"
    )
    args = arg_parser.parse_args()
    config_path = args.config
    config = parse_config(config_path)

    if args.type == 'telegram':
        run_telegram_bot(**config)
    elif args.type == 'console':
        run_console_bot(**config)
    elif args.type == 'dialogue':
        run_dialogue(**config)
    else:
        raise ValueError("Unrecognized conversation type")
