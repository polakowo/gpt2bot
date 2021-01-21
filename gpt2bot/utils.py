import configparser
import logging
import transformers
import numpy as np
import random


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    class ColorCodes:
        grey = "\x1b[38;21m"
        green = "\x1b[1;32m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        blue = "\x1b[1;34m"
        light_blue = "\x1b[1;36m"
        purple = "\x1b[1;35m"
        reset = "\x1b[0m"

    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: ColorCodes.grey + format + ColorCodes.reset,
        logging.INFO: ColorCodes.green + format + ColorCodes.reset,
        logging.WARNING: ColorCodes.yellow + format + ColorCodes.reset,
        logging.ERROR: ColorCodes.red + format + ColorCodes.reset,
        logging.CRITICAL: ColorCodes.bold_red + format + ColorCodes.reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name):
    """Set up logger."""
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    return logger


# Set up logging
transformers.logging.set_verbosity_error()

logger = setup_logger(__name__)


def set_seed(seed):
    """Set seed globally."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except:
        pass


def parse_optional_int(config, section, option):
    value = config.get(section, option)
    return int(value) if value is not None else None


def parse_optional_int_list(config, section, option):
    value = config.get(section, option)
    return list(map(int, value.replace(' ', '').split(','))) if value is not None else None


def parse_config(config_path):
    """Parse config into a dict."""
    logger.info("Parsing the config...")

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    with open(config_path) as f:
        config.read_file(f)

    return dict(
        generator_pipeline=dict(
            model=config.get('generator_pipeline', 'model'),
            config=config.get('generator_pipeline', 'config'),
            tokenizer=config.get('generator_pipeline', 'tokenizer'),
            framework=config.get('generator_pipeline', 'framework'),
            device=config.getint('generator_pipeline', 'device')
        ),
        generator=dict(
            seed=parse_optional_int(config, 'generator', 'seed'),
            max_length=config.getint('generator', 'max_length'),
            min_length=config.getint('generator', 'min_length'),
            do_sample=config.getboolean('generator', 'do_sample'),
            early_stopping=config.getboolean('generator', 'early_stopping'),
            num_beams=config.getint('generator', 'num_beams'),
            temperature=config.getfloat('generator', 'temperature'),
            top_k=config.getint('generator', 'top_k'),
            top_p=config.getfloat('generator', 'top_p'),
            repetition_penalty=config.getfloat('generator', 'repetition_penalty'),
            pad_token_id=parse_optional_int(config, 'generator', 'pad_token_id'),
            bos_token_id=parse_optional_int(config, 'generator', 'bos_token_id'),
            eos_token_id=parse_optional_int(config, 'generator', 'eos_token_id'),
            length_penalty=config.getfloat('generator', 'length_penalty'),
            no_repeat_ngram_size=config.getint('generator', 'no_repeat_ngram_size'),
            bad_words_ids=parse_optional_int_list(config, 'generator', 'bad_words_ids'),
            num_return_sequences=config.getint('generator', 'num_return_sequences'),
            decoder_start_token_id=parse_optional_int(config, 'generator', 'decoder_start_token_id'),
            use_cache=config.getboolean('generator', 'use_cache'),
            clean_up_tokenization_spaces=config.getboolean('generator', 'clean_up_tokenization_spaces')
        ),
        classifier_pipeline=dict(
            model=config.get('classifier_pipeline', 'model'),
            config=config.get('classifier_pipeline', 'config'),
            tokenizer=config.get('classifier_pipeline', 'tokenizer'),
            framework=config.get('classifier_pipeline', 'framework'),
            device=config.getint('classifier_pipeline', 'device')
        ),
        classifier=dict(
            prepend_context=config.getboolean('classifier', 'prepend_context'),
            larger_is_better=config.getboolean('classifier', 'larger_is_better')
        ),
        chatbot=dict(
            max_turns_history=config.getint('chatbot', 'max_turns_history'),
            telegram_token=config.get('chatbot', 'telegram_token'),
            giphy_token=config.get('chatbot', 'giphy_token'),
            giphy_prob=config.getfloat('chatbot', 'giphy_prob'),
            giphy_max_words=config.getint('chatbot', 'giphy_max_words'),
            giphy_weirdness=config.getint('chatbot', 'giphy_weirdness')
        )
    )


def load_generator_pipeline(**kwargs):
    """Load text generation pipeline."""
    logger.info(f"Loading the text generation pipeline '{kwargs.get('model')}'...")

    return transformers.pipeline('text-generation', **kwargs)


def clean_text(txt):
    """Remove unnecessary spaces."""
    return ' '.join(txt.strip().split())


def generate_text(prompt, pipeline, **kwargs):
    """Generate text using pipeline given prompt and other parameters."""
    kwargs = kwargs.copy()

    # Make answers reproducible only if wanted
    seed = kwargs.pop('seed', None)
    if seed is not None:
        set_seed(seed)

    responses = pipeline(prompt, **kwargs)
    return list(map(lambda x: clean_text(x['generated_text'][len(prompt):]), responses))


def load_classifier_pipeline(**kwargs):
    """Load text classification pipeline."""
    logger.info(f"Loading the text classification pipeline '{kwargs.get('model')}'...")

    return transformers.pipeline('sentiment-analysis', **kwargs)


def classify_responses(prompt, responses, pipeline, **kwargs):
    """Classify responses using pipeline given prompt and other parameters."""
    kwargs = kwargs.copy()

    prepend_context = kwargs.pop('prepend_context', True)
    if 'larger_is_better' in kwargs:
        del kwargs['larger_is_better']

    if prepend_context:
        responses = [prompt + pipeline.tokenizer.eos_token + response for response in responses]

    return [output['score'] for output in pipeline(responses)]
