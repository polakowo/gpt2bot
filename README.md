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

gpt2bot implements 
  - a decoder ([source](https://github.com/polakowo/gpt2bot/blob/master/gpt2bot/decoder.py)) for [DialoGPT](https://github.com/microsoft/DialoGPT), 
  - an interactive multiturn chatbot ([source](https://github.com/polakowo/gpt2bot/blob/master/gpt2bot/interactive_bot.py)), and 
  - a Telegram chatbot ([source](https://github.com/polakowo/gpt2bot/blob/master/gpt2bot/telegram_bot.py)).
  
The bot is built around [DialoGPT](https://github.com/microsoft/DialoGPT) - a large-scale pretrained dialogue response generation model trained by Microsoft, which was trained on 147M multi-turn dialogue from Reddit discussion thread. The human evaluation results indicate that its quility is comparable to human response quality under a single-turn conversation Turing test.

Since even with properly filtered Reddit dataset the model can generate toxic/inappropriate responses, the Microsoft team was unable to provide the decoding script. This repository implements the decoding script inspired by `run_generation.py` released earlier by Hugging Face. Moreover, it implements a Telegram bot that can be deployed locally, remotely, and even on Colab, and just makes testing fun.
  
## How to use?

### 1. Create a Telegram bot

- Register a new Telegram bot via BotFather (see https://core.telegram.org/bots)

### 2. Deploy the bot

#### Docker

- Clone the repository
- Set your parameters such as API token in dialog.cfg
- To avoid re-downloading model files at each re-deployment, download the model files beforehand with
```
# cd gpt2bot/gpt2bot
python model.py
```
- Finally, deploy the container from the root folder
```
docker build -t gpt2bot . && docker run gpt2bot
```

#### Google Colab

Inference code can be run on CPU, but it would be slow. A good thing about Google Colab is free GPU. So why not running the Telegram bot there, for blazingly fast chat? Run the notebook at daytime and do not forget to stop it at night.

[A Colab interactive notebook](https://colab.research.google.com/github/polakowo/gpt2bot/blob/master/Demo.ipynb)

#### Manually

- Clone the repository
- Set your parameters such as API token in dialog.cfg
- Install packages listed in requirements.txt
- Run the script
```
# cd gpt2bot/gpt2bot
python telegram_bot.py
```
- To test the things out in the console, run
```
python interactive_bot.py
```

### 3. Start chatting!

![](telegram_bot.gif)

Just start texting. Append @gif for the bot to generate a GIF instead of text. To reset, type "Bye".

## References

- [Official DialoGPT implementation](https://github.com/microsoft/DialoGPT) and [DialoGPT paper](https://arxiv.org/abs/1911.00536)
- [Thread on current decoding scripts](https://github.com/microsoft/DialoGPT/issues/3)

You can wait for a full DialoGPT release and then replace the decoder.
