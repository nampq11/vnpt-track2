from src.brain.llm.services.type import LLMService, LLMServiceConfig
import json
import asyncio
from loguru import logger
from pathlib import Path
from typing import Literal, Dict, List, Optional, Any
from dotenv import load_dotenv
import aiohttp
from src.brain.llm.services.retry_utils import retry_async

load_dotenv()

# Model type mapping to config index
MODEL_CONFIG_MAP = {
    "embedding": 0,
    "small": 1,
    "large": 2,
}

# Cache for config
_CONFIG_CACHE: Dict[str, Any] = {}

def load_config_from_file(
    config_path: str = "config/api-keys.json",
    model_type: Literal["embedding", "small", "large"] = "embedding",
) -> dict:
    """Load VNPT credentials from JSON config file with caching."""
    cache_key = f"{config_path}:{model_type}"
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    full_path = project_root / config_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")
    
    with open(full_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    
    config_index = MODEL_CONFIG_MAP.get(model_type, 0)
    config = configs[config_index]
    
    result = {
        "authorization": config["authorization"],
        "token_id": config["tokenId"],
        "token_key": config["tokenKey"],
        "api_name": config["llmApiName"],
    }
    
    _CONFIG_CACHE[cache_key] = result
    return result


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
        self.provider = "vnpt"  # Explicitly set provider


class VNPTService(LLMService):

    def __init__(
        self,
        base_url: str = "https://api.idg.vnpt.vn",
        model: str = "vnptai-hackathon-small",
        model_type: Literal["embedding", "small", "large"] = "small",
        config_path: str = "config/api-keys.json",
        verbose: bool = False
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.model_type = model_type
        self.verbose = verbose
        
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
        system_message: Optional[str] = None,
        stream: Optional[bool] = False,
        verbose: bool = False
    ) -> str:
        """Generate response from VNPT AI API with retry logic (3 retries, exponential backoff)"""
        if stream:
            raise NotImplementedError("Streaming is not supported by VNPTService yet.")
        
        try:
            return await self._generate_with_retry(user_input, system_message, verbose)
        except Exception as e:
            raise RuntimeError(f"Error generating response from VNPT: {str(e)}")
    
    @retry_async(
        max_retries=3,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)
    )
    async def _generate_with_retry(self, user_input: str, system_message: Optional[str] = None, verbose: bool = False) -> str:
        """Internal method with retry logic for generate()"""
        headers = self._get_headers()
        
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': user_input})

        if verbose or self.verbose:
            logger.info(f"Sending request to VNPT with model={self.model}, messages={str(messages)}")

        json_data = {
            'model': self.model.replace('-', '_'),
            'messages': messages,
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': 20,
            'n': 1,
            'max_completion_tokens': 1000,
        }
        
        # Create a new session for each request (simple, but creating session is somewhat expensive)
        # Ideally, we should pass session or manage it in the class. 
        # For now, using Context Manager is safe and clean.
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url + '/data-service/v1/chat/completions/' + self.model,
                headers=headers,
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data['choices'][0]['message']['content']
    
    async def get_embedding(
        self,
        session: aiohttp.ClientSession,
        text: str,
    ) -> List[float]:
        """Get embedding from VNPT AI API with retry logic (3 retries, exponential backoff)"""
        try:
            return await self._get_embedding_with_retry(session, text)
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from VNPT: {str(e)}")
    
    @retry_async(
        max_retries=3,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)
    )
    async def _get_embedding_with_retry(
        self,
        session: aiohttp.ClientSession,
        text: str,
    ) -> List[float]:
        """Internal method with retry logic for get_embedding()"""
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
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            # Parse embedding here
            if isinstance(data, dict) and 'data' in data:
                return data['data'][0]['embedding']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Unexpected embedding format: {data}")
    
    def get_all_tools(self) -> dict:
        return {}
    
    def get_config(self) -> VNPTServiceConfig:
        return self.config
