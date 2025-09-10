import os
import yaml

# Paths to essential directories
MODEL_PATH = "/path/to/model"
DATA_PATH = "/path/to/data"
OUTPUT_PATH = "/path/to/output"
EXPERIMENTS_PATH = "/path/to/experiments"
LOG_PATH = "/path/to/logs"
EMBEDDINGS_PATH = "/path/to/embeddings"
# Paths to essential files

class Settings:
    def __init__(self, config_path):
        self.MODEL_PATH = MODEL_PATH
        self.DATA_PATH = DATA_PATH
        self.OUTPUT_PATH = OUTPUT_PATH
        self.EXPERIMENTS_PATH = EXPERIMENTS_PATH
        self.LOG_PATH = LOG_PATH
        self.config = self.load_config(config_path)

        self.model = self.config.get('model', {})
        self.settings = self.config.get('settings', {})

    def __iter__(self):
        return iter((self.model, self.settings))
    
    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
# Comment test
