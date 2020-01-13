# GPT2Bot

GPT2Bot implements 
  - a decoder ([source](https://github.com/polakowo/gpt2bot/blob/master/decoder.py)) for [DialoGPT](https://github.com/microsoft/DialoGPT), 
  - an interactive multiturn chatbot ([source](https://github.com/polakowo/gpt2bot/blob/master/interactive_bot.py)), and 
  - a Telegram chatbot ([source](https://github.com/polakowo/gpt2bot/blob/master/telegram_bot.py)).
  
## How to use?

### Docker deployment

- Set your parameters in dialog.cfg
- If you run the container locally, to avoid downloading model files during each deployment, it is advised to download the model files first. To do this, run
```
python model.py
```
- Finally, deploy the container by using
```
docker build -t gpt2bot . && docker run gpt2bot
```

### Google Colab

A good thing about Google Colab is free GPU. So why not running the Telegram bot there, for blazingly fast chat? Run the notebook at daytime and do not forget to stop it at night.

[A Colab interactive notebook](https://colab.research.google.com/github/polakowo/gpt2bot/blob/master/Demo.ipynb)

### Run manually

- Requires 
  - [pytorch](https://github.com/pytorch/pytorch) (tested on 1.2.0), 
  - [transformers](https://github.com/huggingface/transformers) (tested on 2.3.0), and 
  - [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) (tested on 12.3.0)

#### Interactive chatbot

- Set your parameters in dialog.cfg
- Run `python interactive_bot.py`

An example of a fun dialog:
```
User >>> Can we achieve singularity?
Bot >>> What does this mean?
User >>> Can computers become smarter than humans?
Bot >>> Is there any evidence that this is possible?
User >>> It must be doable
Bot >>> But... how?
User >>> I want to find out, that's why I created you
Bot >>> You created me?
User >>> You're my bot
Bot >>> You monster
```

#### Telegram chatbot

The same as the interactive chatbot but in Telegram and supports gifs.

- Create a new Telegram bot via BotFather (see https://core.telegram.org/bots)
- Set your parameters such as API token in dialog.cfg
- Run `python telegram_bot.py`

![](telegram_bot.gif)
