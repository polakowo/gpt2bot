from .utils import *
import discord
import time
import random
from dotenv import load_dotenv
import os

load_dotenv()


class MyClient(discord.Client):

    channel_name = os.environ.get("CHANNEL_NAME")  # channel to work on
    min_time = os.environ.get("DELAY")  # minimum time to elapse to respond again

    def __init__(self, botObj):
        super().__init__()
        self.bot = botObj
        self.last_time = time.time()
        self.rec_messages = []

    def can_send(self):
        elapsed = time.time() - self.last_time
        if elapsed > self.min_time:
            self.last_time = time.time()
            return True
        else:
            return False

    def check_mention(self, m):
        if m.reference is not None and not m.is_system:
            return True
        return False

    async def on_ready(self):
        print('Logged on as', self.user)

    async def on_message(self, message):
        # Only work on one channel
        if self.channel_name in message.channel.name and not '2' in message.channel.name:

            # don't respond to ourselves
            if message.author == self.user:
                return

            # Append message to possible messages to answer list
            self.rec_messages.append(message.content)

            # Check if the bot it mentioned
            mention = f'<@!{self.user.id}>'
            if mention in message.content:
                print(f'BOT MENTION: {message.content}')

            # Ping command
            if message.content == 'qqqq':
                await message.channel.send('qqqq?')

            else:
                if self.can_send() and len(self.rec_messages) != 0:  # If it is time to send a message and there are messages to respond to
                    async with message.channel.typing():
                        filtered = list(filter(lambda x: len(
                            x) < 900 and '\n' not in x, self.rec_messages))  # filter long messages away  

                        # Pick random message to answer
                        to_answer = random.choice(filtered)
                        self.rec_messages.clear()  # Clear the rec_messages
                        print(f'Answering: {to_answer}')
                        response = self.bot.gen_message(to_answer)
                        await message.channel.send(response)


# //////

class DiscordBot:

    def __init__(self, **kwargs):
        # Extract parameters
        general_params = kwargs.get('general_params', {})
        device = general_params.get('device', -1)
        seed = general_params.get('seed', None)
        debug = general_params.get('debug', False)

        generation_pipeline_kwargs = kwargs.get(
            'generation_pipeline_kwargs', {})
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

        self.generation_pipeline_kwargs = generation_pipeline_kwargs
        self.generator_kwargs = generator_kwargs
        self.prior_ranker_weights = prior_ranker_weights
        self.cond_ranker_weights = cond_ranker_weights
        self.chatbot_params = chatbot_params
        self.device = device
        self.seed = seed
        self.debug = debug

        self.turns = []

        # Prepare the pipelines
        self.generation_pipeline = load_pipeline(
            'text-generation', device=device, **generation_pipeline_kwargs)
        self.ranker_dict = build_ranker_dict(
            device=device, **prior_ranker_weights, **cond_ranker_weights)

    def gen_message(self, msg):
        """Receive message, generate response, and send it back to the user."""

        max_turns_history = self.chatbot_params.get('max_turns_history', 2)

        user_message = msg

        if max_turns_history == 0:
            self.turns = []
        # A single turn is a group of user messages and bot responses right after
        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        self.turns.append(turn)
        turn['user_messages'].append(user_message)
        logger.debug(f"User: {user_message}")
        # Merge turns into a single prompt (don't forget EOS token)
        prompt = ""
        from_index = max(len(self.turns) - max_turns_history - 1,
                         0) if max_turns_history >= 0 else 0
        for turn in self.turns[from_index:]:
            # Each turn begins with user messages
            for user_message in turn['user_messages']:
                prompt += clean_text(user_message) + \
                    self.generation_pipeline.tokenizer.eos_token
            for bot_message in turn['bot_messages']:
                prompt += clean_text(bot_message) + \
                    self.generation_pipeline.tokenizer.eos_token

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
        logger.debug(f"Bot: {bot_message}")
        return bot_message

    def run(self):
        """Run the chatbot."""
        logger.info("Running the discord bot...")
        client = MyClient(self)
        client.run(os.environ.get("DISCORD_TOKEN"))


def run(**kwargs):
    """Run `TelegramBot`."""
    DiscordBot(**kwargs).run()
