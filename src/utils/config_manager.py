# src/utils/config_manager.py

import os
from typing import Any, Dict
from omegaconf import OmegaConf
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self._load_env_vars()

    def _load_env_vars(self):
        env_path = os.path.join(os.path.dirname(__file__), '../../config/secrets.env')
        load_dotenv(env_path)

    def get(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self.config, key, default=default)

    def get_exchange_credentials(self) -> Dict[str, str]:
        return {
            'api_key': os.getenv(f"{self.get('exchange.name').upper()}_ACCESS_KEY"),
            'api_secret': os.getenv(f"{self.get('exchange.name').upper()}_SECRET_KEY")
        }