# !pip install python-telegram-bot --upgrade
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction, ParseMode
from functools import wraps
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .utils import *

logger = setup_logger(__name__)


def start_command(update, context):
    """Start a new dialogue with this message."""

    context.chat_data['turns'] = []
    update.message.reply_text("Just start texting me. "
                              "Append \"@gif\" for me to generate a GIF. "
                              "If I'm getting annoying, type \"/start\". "
                              "Make sure to send no more than one message at a time.")


def requests_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504), session=None):
    """Retry n times if unsuccessful."""

    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def translate_message_to_gif(message, **chatbot_kwargs):
    """Translate message text into a GIF.

    See https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/"""

    params = {
        'api_key': chatbot_kwargs['giphy_token'],
        's': message,
        'weirdness': chatbot_kwargs.get('giphy_weirdness', 5)
    }
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    response = requests_retry_session().get(url)
    return response.json()['data']['images']['fixed_height']['url']


def self_decorator(self, func):
    """Passes bot object to func command."""

    # TODO: Any other ways to pass variables to handlers?
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)

    return command_func


def send_action(action):
    """Sends `action` while processing func command."""

    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)

        return command_func

    return decorator


send_typing_action = send_action(ChatAction.TYPING)


@send_typing_action
def message(self, update, context):
    """Receive message, generate response, and send it back to the user."""

    max_turns_history = self.chatbot_kwargs.get('max_turns_history', 2)
    giphy_prob = self.chatbot_kwargs.get('giphy_prob', 0.1)
    giphy_max_words = self.chatbot_kwargs.get('giphy_max_words', 10)

    if 'turns' not in context.chat_data:
        context.chat_data['turns'] = []
    turns = context.chat_data['turns']

    user_message = update.message.text
    return_gif = False
    if '@gif' in user_message:
        # Return gif
        return_gif = True
        user_message = user_message.replace('@gif', '').strip()
    if max_turns_history == 0:
        context.chat_data['turns'] = []
    # A single turn is a group of user messages and bot responses right after
    turn = {
        'user_messages': [],
        'bot_messages': []
    }
    turns.append(turn)
    turn['user_messages'].append(user_message)
    logger.debug(f"{update.effective_message.chat_id} - User >>> {user_message}")
    # Merge turns into a single prompt (don't forget EOS token)
    prompt = ""
    from_index = max(len(turns) - max_turns_history - 1, 0) if max_turns_history >= 0 else 0
    for turn in turns[from_index:]:
        # Each turn begins with user messages
        for user_message in turn['user_messages']:
            prompt += clean_text(user_message) + self.generation_pipeline.tokenizer.eos_token
        for bot_message in turn['bot_messages']:
            prompt += clean_text(bot_message) + self.generation_pipeline.tokenizer.eos_token

    # Generate bot messages
    bot_messages = generate_responses(
        prompt,
        self.generation_pipeline,
        seed=self.seed,
        debug=self.debug,
        **self.generator_kwargs
    )
    if len(bot_messages) == 1:
        bot_message = bot_messages[0]
    else:
        bot_message = pick_best_response(
            prompt,
            bot_messages,
            self.ranker_dict,
            debug=self.debug
        )
    turn['bot_messages'].append(bot_message)
    logger.debug(f"{update.effective_message.chat_id} - Bot >>> {bot_message}")
    # Return response as text
    update.message.reply_text(bot_message)
    if len(bot_message.split()) <= giphy_max_words and random.random() < giphy_prob:
        return_gif = True
    if return_gif:
        # Also return the response as a GIF
        gif_url = translate_message_to_gif(bot_message, **self.chatbot_kwargs)
        context.bot.send_animation(update.effective_message.chat_id, gif_url)


def error(update, context):
    logger.warning(context.error)


class TelegramBot:
    """Runs the Telegram bot based on python-telegram-bot.

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

    def __init__(
            self,
            generation_pipeline_kwargs={},
            generator_kwargs={},
            prior_rankers_kwargs={},
            cond_rankers_kwargs={},
            chatbot_kwargs={},
            device=-1,
            seed=None,
            debug=False
    ):
        self.generation_pipeline_kwargs = generation_pipeline_kwargs
        self.generator_kwargs = generator_kwargs
        self.prior_rankers_kwargs = prior_rankers_kwargs
        self.cond_rankers_kwargs = cond_rankers_kwargs
        self.chatbot_kwargs = chatbot_kwargs
        self.device = device
        self.seed = seed
        self.debug = debug

        # Prepare the pipelines
        self.generation_pipeline = load_pipeline('text-generation', device=device, **generation_pipeline_kwargs)
        self.ranker_dict = build_ranker_dict(device=device, **prior_rankers_kwargs, **cond_rankers_kwargs)

        # Initialize the chatbot
        logger.info("Initializing the chatbot...")
        self.updater = Updater(chatbot_kwargs['telegram_token'], use_context=True)

        # Add command, message and error handlers
        dp = self.updater.dispatcher
        dp.add_handler(CommandHandler('start', start_command))
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))
        dp.add_error_handler(error)

    def run_bot(self):
        """Run the chatbot."""
        logger.info("Running the chatbot...")

        # Start the Bot
        self.updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()
