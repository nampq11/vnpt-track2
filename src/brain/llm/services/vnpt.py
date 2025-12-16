from src.brain.llm.services.type import LLMService, LLMServiceConfig
import os
import json
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import requests
import aiohttp

load_dotenv()

# Model type mapping to config index
MODEL_CONFIG_MAP = {
    "embedding": 0,
    "small": 1,
    "large": 2,
}

def load_config_from_file(
    config_path: str = "config/vnpt.json",
    model_type: Literal["embedding", "small", "large"] = "embedding",
) -> dict:
    """Load VNPT credentials from JSON config file."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    with open(full_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    config_index = MODEL_CONFIG_MAP.get(model_type, 0)
    config = configs[config_index]
    
    return {
        "authorization": config["authorization"],
        "token_id": config["tokenId"],
        "token_key": config["tokenKey"],
        "api_name": config["llmApiName"],
    }


class VNPTServiceConfig(LLMServiceConfig):
    def __init__(
        self,
        authorization: str,
        token_id: str,
        token_key: str,
        model: str,
    ) -> None:
        self.authorization = authorization
        self.token_id = token_id
        self.token_key = token_key
        self.model = model


class VNPTService(LLMService):

    def __init__(
        self,
        base_url: str = "https://api.idg.vnpt.vn",
        model: str = "vnptai-hackathon-small",
        model_type: Literal["embedding", "small", "large"] = "small",
        config_path: str = "config/vnpt.json",
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.model_type = model_type
        
        # Load config from JSON file
        config_data = load_config_from_file(config_path, model_type)
        
        self.config = VNPTServiceConfig(
            authorization=config_data["authorization"],
            token_id=config_data["token_id"],
            token_key=config_data["token_key"],
            model=model
        )

    def _get_headers(self, model_type: Literal["embedding", "small", "large"] = None) -> dict:
        """Get headers for API request, optionally for a different model type."""
        if model_type and model_type != self.model_type:
            config_data = load_config_from_file(model_type=model_type)
            return {
                'Authorization': config_data["authorization"],
                'Token-id': config_data["token_id"],
                'Token-key': config_data["token_key"],
                'Content-Type': 'application/json',
            }
        return {
            'Authorization': self.config.authorization,
            'Token-id': self.config.token_id,
            'Token-key': self.config.token_key,
            'Content-Type': 'application/json',
        }

    async def generate(
        self,
        user_input: str,
    ) -> str:
        try:
            headers = self._get_headers()
            json_data = {
                'model': 'vnptai_hackathon_small',
                'messages': [
                    {'role': 'user', 'content': user_input}
                ],
                'temperature': 1.0,
                'top_p': 1.0,
                'top_k': 20,
                'n': 1,
                'max_completion_tokens': 1000,
            }
            response = requests.post(self.base_url + '/data-service/v1/chat/completions/' + self.model, headers=headers, json=json_data)
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"Error generating response from VNPT: {str(e)}")
    
    async def get_embedding(
        self,
        session: aiohttp.ClientSession,
        text: str,
    ):
        try:
            # Always use embedding config for embeddings
            headers = self._get_headers(model_type="embedding")
            json_data = {
                'model': 'vnptai_hackathon_embedding',
                'input': text,
                'encoding_format': 'float',
            }
            async with session.post(
                self.base_url + '/data-service/vnptai-hackathon-embedding',
                headers=headers,
                json=json_data,
            ) as response:
                return await response.json()
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from VNPT: {str(e)}")
    
    def get_all_tools(self) -> dict:
        return {}
    
    def get_config(self) -> VNPTServiceConfig:
        return self.config
