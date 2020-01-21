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

# Model configuration files
CONFIG_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/config.json',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/config.json'
}
VOCAB_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/vocab.json',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/vocab.json'
}
MERGE_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/merges.txt',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/merges.txt'
}

# Model files
# Note that the model size is roughly half of the GPT model because our model is saved by fp16
LSP_MODEL_URL = {
    'multiref': {
        'medium_fs': 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_fs.pkl',
        'medium_ft': 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl',
        'small_fs': 'https://convaisharables.blob.core.windows.net/lsp/multiref/small_fs.pkl',
        'small_ft': 'https://convaisharables.blob.core.windows.net/lsp/multiref/small_ft.pkl'
    },
    'dstc': { # medium_ft.pkl is actually a small model
        'small_ft': 'https://convaisharables.blob.core.windows.net/lsp/DSTC/medium_ft.pkl'
    }
}

# The reverse model is predicting the source from the target. This model is used for MMI reranking.
# small_reverse.pkl is actually a medium model
REVERSE_MODEL_URL = 'https://convaisharables.blob.core.windows.net/lsp/multiref/small_reverse.pkl'

def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    file_name = os.path.basename(url)
    if 'pytorch_model.bin' in file_name:
        file_name = 'pytorch_model.bin'

    if os.path.isfile(os.path.join(folder, file_name)):
        return

    with open(os.path.join(folder, file_name), 'wb') as f:
        http_get(url, f)


def download_model_folder(config):
    # Parse parameters
    data_folder = config.get('model', 'data_folder')
    model_size = config.get('model', 'model_size')
    dataset = config.get('model', 'dataset')
    from_scratch = config.getboolean('model', 'from_scratch')

    # Create data folder if needed
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    # Build target folder name (must be unique across all parameter combinations)
    target_folder_name = model_size + "_" + dataset + ("_fs" if from_scratch else "_ft")
    target_folder = os.path.join(data_folder, target_folder_name)
    # Download files
    logger.info(f"Downloading model files to {target_folder_name}...")
    download_file(CONFIG_FILE[model_size], target_folder)
    download_file(VOCAB_FILE[model_size], target_folder)
    download_file(MERGE_FILE[model_size], target_folder)
    model_train_type = model_size + ('_fs' if from_scratch else '_ft')
    if model_train_type not in LSP_MODEL_URL[dataset]:
        k = ','.join(list(LSP_MODEL_URL[dataset].keys()))
        raise ValueError(f"'{model_train_type}' not exist for dataset '{dataset}', please choose from [{k}]")
    download_file(LSP_MODEL_URL[dataset][model_train_type], target_folder)
    return target_folder_name

def download_reverse_model_folder(config):
    # Parse parameters
    data_folder = config.get('model', 'data_folder')
    # Only one size is currently supported
    model_size = 'medium'

    # Create data folder if needed
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    # Build target folder name (must be unique across all parameter combinations)
    target_folder_name = model_size + '_reverse'
    target_folder = os.path.join(data_folder, target_folder_name)
    # Download files
    logger.info(f"Downloading model files to {target_folder_name}...")
    download_file(CONFIG_FILE[model_size], target_folder)
    download_file(VOCAB_FILE[model_size], target_folder)
    download_file(MERGE_FILE[model_size], target_folder)
    download_file(REVERSE_MODEL_URL, target_folder)
    return target_folder_name

def load_model(target_folder_name, config):
    # Parse parameters
    data_folder = config.get('model', 'data_folder')
    model_size = config.get('model', 'model_size')
    no_cuda = config.getboolean('model', 'no_cuda')

    logger.info(f"Loading model from {target_folder_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    # Tokenizer
    target_folder = os.path.join(data_folder, target_folder_name)
    tokenizer = GPT2Tokenizer(os.path.join(target_folder, 'vocab.json'), os.path.join(target_folder, 'merges.txt'))
    # Config
    config = GPT2Config.from_json_file(os.path.join(target_folder, 'config.json'))
    # Weights
    state_dict_path = glob(os.path.join(target_folder, f'*.pkl'))[0]
    state_dict = torch.load(state_dict_path, map_location=device)
    if model_size == 'small':
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    state_dict['lm_head.weight'] = state_dict['lm_head.decoder.weight']
    state_dict.pop("lm_head.decoder.weight", None)
    # Model
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

def main():
    # Script arguments can include path of the config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default="chatbot.cfg")
    args = arg_parser.parse_args()

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    with open(args.config) as f:
        config.read_file(f)

    # Download main model
    download_model_folder(config)
    # Download reverse model
    use_mmi = config.getboolean('model', 'use_mmi')
    if use_mmi:
        download_reverse_model_folder(config)
    

if __name__ == '__main__':
    main()