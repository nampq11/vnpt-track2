"""VNPT API service implementation (for Docker runtime)

Based on VNPT API Documentation:
- Chat Completion Endpoints:
  - Small: https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small
  - Large: https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-large
- Embedding: https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding
- Auth Headers: Authorization (Bearer token), Token-id, Token-key
"""

from typing import Optional, List, Dict, Any
import aiohttp
import asyncio
from .base import LLMService, LLMServiceConfig, LLMServiceResponse
from src.core.config import VNPTConfig


class VNPTService(LLMService):
    """VNPT API client for Docker runtime
    
    Supports both small and large model endpoints with configurable parameters.
    Implements VNPT hackathon API specification with proper authentication.
    """
    
    # API Configuration from VNPT Documentation
    API_BASE_URL = "https://api.idg.vnpt.vn"
    CHAT_ENDPOINTS = {
        "small": "/data-service/v1/chat/completions/vnptai-hackathon-small",
        "large": "/data-service/v1/chat/completions/vnptai-hackathon-large",
    }
    EMBEDDING_ENDPOINT = "/data-service/vnptai-hackathon-embedding"
    
    # Model names from VNPT API
    MODEL_NAMES = {
        "small": "vnptai_hackathon_small",
        "large": "vnptai_hackathon_large",
        "embedding": "vnptai_hackathon_embedding",
    }
    
    # Quota limits from API documentation
    QUOTAS = {
        "small": {"req_per_day": 1000, "req_per_hour": 60},
        "large": {"req_per_day": 500, "req_per_hour": 40},
        "embedding": {"req_per_minute": 500},
    }
    
    def __init__(self, config: VNPTConfig, model_size: str = "small"):
        """
        Initialize VNPT service
        
        Args:
            config: VNPTConfig with API keys and tokens
            model_size: "small" or "large" model variant
        """
        self.config_obj = config
        self.model_size = model_size
        self.config = LLMServiceConfig(
            provider="vnpt", 
            model=self.MODEL_NAMES.get(model_size, "vnptai_hackathon_small")
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate required configuration for the specific model"""
        credentials = self.config_obj.get_credentials(self.model_size)
        if not credentials or not credentials.api_key or not credentials.token_id or not credentials.token_key:
            raise ValueError(
                f"VNPT credentials not configured for model '{self.model_size}'. "
                f"Please set VNPT_CONFIG_FILE or VNPT_API_KEY_* environment variables."
            )
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get VNPT API authentication headers per API documentation"""
        credentials = self.config_obj.get_credentials(self.model_size)
        if not credentials:
            raise ValueError(f"No credentials found for model '{self.model_size}'")
        
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Token-id": credentials.token_id,
            "Token-key": credentials.token_key,
            "Content-Type": "application/json",
        }
    
    async def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        top_k: int = 20,
        n: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> LLMServiceResponse:
        """
        Generate response via VNPT Chat API
        
        Args:
            user_input: User query
            system_prompt: System message
            temperature: Sampling temperature (0.0-2.0, default 1.0)
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling threshold (0.0-1.0, default 1.0)
            top_k: Top K tokens to consider (default 20)
            n: Number of completions (default 1)
            presence_penalty: Penalty for token presence (-2.0 to 2.0, default 0.0)
            frequency_penalty: Penalty for token frequency (-2.0 to 2.0, default 0.0)
            
        Returns:
            LLMServiceResponse
            
        Raises:
            ValueError: If configuration is missing
            RuntimeError: If API call fails
        """
        session = await self._ensure_session()
        
        # Build request payload per VNPT API spec
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        
        payload = {
            "model": self.MODEL_NAMES.get(self.model_size, "vnptai_hackathon_small"),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "n": n,
            "max_completion_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        endpoint = self.API_BASE_URL + self.CHAT_ENDPOINTS[self.model_size]
        headers = self._get_auth_headers()
        
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config_obj.request_timeout),
            ) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise RuntimeError(
                        f"VNPT API error {resp.status}: {error_detail}"
                    )
                
                data = await resp.json()
                
                # Extract content from response per VNPT API spec
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices in VNPT API response")
                
                content = choices[0].get("message", {}).get("content", "")
                
                return LLMServiceResponse(
                    content=content,
                    usage=data.get("usage"),
                )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"VNPT API request timeout after {self.config_obj.request_timeout}s"
            )
        except Exception as e:
            raise RuntimeError(f"VNPT API error: {str(e)}")
    
    async def embed(self, text: str, encoding_format: str = "float") -> List[float]:
        """
        Get embedding from VNPT Embedding API
        
        Args:
            text: Text to embed (Vietnamese text supported)
            encoding_format: Format of embedding ("float" or "base64", default "float")
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If configuration is missing
            RuntimeError: If API call fails
        """
        session = await self._ensure_session()
        
        # Build request payload per VNPT Embedding API spec (Section 3.3)
        payload = {
            "model": self.MODEL_NAMES["embedding"],
            "input": text,
            "encoding_format": encoding_format,
        }
        
        endpoint = self.API_BASE_URL + self.EMBEDDING_ENDPOINT
        headers = self._get_auth_headers()
        
        try:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.config_obj.request_timeout),
            ) as resp:
                if resp.status != 200:
                    error_detail = await resp.text()
                    raise RuntimeError(
                        f"VNPT Embedding API error {resp.status}: {error_detail}"
                    )
                
                data = await resp.json()
                
                # Extract embedding from response per VNPT API spec
                # Response structure: {"data": [{"embedding": [...], "index": 0}], ...}
                data_list = data.get("data", [])
                if not data_list:
                    raise RuntimeError("No embedding data in VNPT API response")
                
                embedding = data_list[0].get("embedding", [])
                
                return embedding
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"VNPT Embedding API request timeout after {self.config_obj.request_timeout}s"
            )
        except Exception as e:
            raise RuntimeError(f"VNPT Embedding API error: {str(e)}")
    
    def get_config(self) -> LLMServiceConfig:
        """Get service configuration"""
        return self.config
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()

