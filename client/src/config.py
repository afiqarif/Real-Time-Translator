# config.py
import json
import os

LANG_CODES = {
    "Japanese": "ja",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Korean": "ko",
    "Mandarin Chinese": "zh",
    "Russian": "ru",
    "English": "en"
}

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "lightning_url": "",
    "source_lang": "en",
    "target_lang": "Japanese"
}

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except:
        return DEFAULT_CONFIG.copy()

def save_config(config_data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f)