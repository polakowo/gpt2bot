# gpt2bot

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

<img src="https://github.com/polakowo/gpt2bot/blob/master/reddit.png?raw=true" width=80>
  
## How to use?

### 1. Create a Telegram bot

- Register a new Telegram bot via BotFather (see https://core.telegram.org/bots)

### 2. Deploy the bot

To get started, first clone this repo:

```
$ git clone https://github.com/polakowo/gpt2bot.git
$ cd gpt2bot
```

Create and activate a conda env:

```
$ conda create -n gpt2bot python=3.7.6
$ conda activate gpt2bot
```

Or a venv (make sure your Python is 3.6+):

```
$ python3 -m venv venv
$ source venv/bin/activate  # Unix
$ venv\Scripts\activate  # Windows
```

Set your parameters such as API token in chatbot.cfg:

```
$ nano chatbot.cfg
```

Install the requirements:

```
$ pip install -r requirements.txt
```

Run the chatbot:

```
$ python run_telegram_bot.py
```

### 3. Start chatting!

![](telegram_bot.gif)

Just start texting. Append @gif for the bot to also generate a GIF. To reset, type "/start".
