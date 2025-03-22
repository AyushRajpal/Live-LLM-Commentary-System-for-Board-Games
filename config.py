# this is to avoid circular imports

# Configuration
# Load configuration from JSON file or create default configuration
import json
import logging
import os


CONFIG_FILE = "./config.json"

# Default configuration
DEFAULT_CONFIG = {
    "llm_provider": "mock",  # Options: "gemini", "openai", "mock" - using mock by default
    "api_key": "",  # Fill in your API key later
    "commentary_style": "educational",  # Options: "educational", "casual", "expert"
    "text_to_speech": True,  # Enabled for testing
    "board_size": 8,
    "window_width": 800,
    "window_height": 600,
}
CONFIG = None

def load_config():
    global CONFIG
    # Try to load from file or use defaults
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                CONFIG = json.load(f)
                logging.info(f"Configuration loaded from {CONFIG_FILE}")
        else:
            CONFIG = DEFAULT_CONFIG
            # Save default config to file for user to edit later
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
                logging.info(f"Default configuration created at {CONFIG_FILE}")
    except Exception as e:
        logging.error(f"Error loading configuration: {str(e)}")
        CONFIG = DEFAULT_CONFIG

def get_config():
    if CONFIG is None:
        load_config()
    return CONFIG