import os
import requests
from tqdm import tqdm
from glob import glob
import torch
import configparser
import argparse
import logging

# !pip install transformers==2.3.0
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# If you get tensorflow deprecation warnings, run
# pip uninstall numpy
# pip install numpy==1.16.4

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(folder_name, config):
    # Parse parameters
    no_cuda = config.getboolean('model', 'no_cuda')

    logger.info(f"Loading model from {folder_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    # Model
    model = GPT2LMHeadModel.from_pretrained(folder_name)
    model.to(device)
    model.eval()
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(folder_name)
    return model, tokenizer