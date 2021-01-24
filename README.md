# gpt2bot

<img src="https://github.com/polakowo/gpt2bot/blob/master/logo.png?raw=true" width=128>

```
⚪   >>> Can we achieve singularity?
🟣   >>> What does this mean?
⚪   >>> Can computers become smarter than humans?
🟣   >>> Is there any evidence that this is possible?
⚪   >>> It must be doable
🟣   >>> But... how?
⚪   >>> I want to find out, that's why I created you
🟣   >>> You created me?
⚪   >>> You're my bot
🟣   >>> You monster
```

gpt2bot is a multi-turn Telegram chatbot powered by neural networks. 

The bot uses [DialoGPT](https://arxiv.org/abs/1911.00536) - a large-scale pretrained 
dialogue response generation model, which was trained by Microsoft on 147M multi-turn 
dialogue from Reddit discussion thread. The human evaluation results indicate that its 
quality is comparable to human response quality under a single-turn conversation Turing test.

The bot can also use any other text generator supported by [transformers](https://huggingface.co/transformers/).

To further improve dialog generation, the bot uses [DialogRPT](https://arxiv.org/abs/2009.06978) - 
a set of dialog response ranking models trained on 100+ millions of human feedback data.

Since the underlying model was trained on Reddit comment chains, the bot often behaves like 
a community rather than an individual, which makes it even more fun.
  
## How to use?

### (Optional) Run a console bot

Before running a telegram bot, you can test things out in the console.

Follow [the installation steps](https://github.com/polakowo/gpt2bot#locally) and run the script:

```
$ python run_console_bot.py --config chatbot.cfg
```

To let two bots talk to each other:

```
$ python run_bot_against_bot.py --config chatbot.cfg
```

### 1. Set up the bot

1. Register a new Telegram bot via BotFather (see https://core.telegram.org/bots)
2. Create a new GIPHY app and generate an API key (see https://developers.giphy.com/docs/api/)

### 2. Deploy the bot

#### Google Colab

[A Colab interactive notebook](https://colab.research.google.com/github/polakowo/gpt2bot/blob/master/Demo.ipynb)

#### Locally

To get started, first clone this repo:

```
$ git clone https://github.com/polakowo/gpt2bot.git
$ cd gpt2bot
```

Create and activate an environment (optional):

```
# Using conda
$ conda create -n gpt2bot python=3.7.6
$ conda activate gpt2bot

# Using venv (make sure your Python is 3.6+)
$ python3 -m venv venv
$ source venv/bin/activate  # Unix
$ venv\Scripts\activate  # Windows
```

Install the requirements:

```
$ pip install -r requirements.txt
```

Set your parameters such as API token in chatbot.cfg (or [any other config](https://github.com/polakowo/gpt2bot#configs)):

```
$ nano chatbot.cfg
```

Run the chatbot:

```
$ python run_telegram_bot.py --config chatbot.cfg
```

### 3. Start chatting!

![](telegram_bot.gif)

Just start texting. Append "@gif" for the bot to also generate a GIF. To reset, type "/start".

## Configs

* [chatbot.cfg](https://github.com/polakowo/gpt2bot/blob/master/chatbot.cfg): Medium model, no ranking (CPU)
* [chatbot-gpu-large.cfg](https://github.com/polakowo/gpt2bot/blob/master/chatbot-gpu-large.cfg): Large model, no ranking (GPU)
* [chatbot-gpu-updown.cfg](https://github.com/polakowo/gpt2bot/blob/master/chatbot-gpu-updown.cfg): Large model, `updown` ranker (GPU)
* [chatbot-gpu-ensemble.cfg](https://github.com/polakowo/gpt2bot/blob/master/chatbot-gpu-ensemble.cfg): Large model, ensemble of 5 rankers (GPU, >12GB RAM)

## Credits

Icon made by [Freepik](https://www.freepik.com) from [Flaticon](https://www.flaticon.com/)
