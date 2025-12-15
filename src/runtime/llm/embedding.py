"""Embedding service interface

Supports VNPT Embedding API with Vietnamese text support.

Based on VNPT API Documentation (Section 3.3):
- Endpoint: https://api.idg.vnpt.vn/data-service/vnptai-hackathon-embedding
- Model: vnptai_hackathon_embedding
- Quota: 500 req/minute
- Auth: Bearer token + Token-id + Token-key
- Input: Text to embed
- Output: Embedding vector (float list)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import aiohttp
import asyncio
from src.core.config import VNPTConfig


class EmbeddingService(ABC):
    """Abstract embedding service interface"""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Get embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class VNPTEmbeddingService(EmbeddingService):
    """VNPT Embedding API client with Vietnamese support
    
    Provides both single and batch embedding with rate limiting.
    Respects VNPT API quota (500 req/minute).
    """
    
    # API Configuration from VNPT Documentation
    API_BASE_URL = "https://api.idg.vnpt.vn"
    EMBEDDING_ENDPOINT = "/data-service/vnptai-hackathon-embedding"
    MODEL_NAME = "vnptai_hackathon_embedding"
    QUOTA_REQ_PER_MINUTE = 500
    
    def __init__(
        self,
        config: VNPTConfig,
        timeout: int = 30,
        max_concurrent: int = 8
    ):
        """
        Initialize VNPT embedding service
        
        Args:
            config: VNPTConfig with API keys and tokens
            timeout: Request timeout in seconds (default 30)
            max_concurrent: Maximum concurrent requests (default 8, respects quota)
        """
        self.config = config
        self.timeout = timeout
        self.max_concurrent = min(max_concurrent, 8)  # Respect rate limits
        self.session: Optional[aiohttp.ClientSession] = None
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate required configuration for embedding model"""
        credentials = self.config.get_credentials("embedding")
        if not credentials or not credentials.api_key or not credentials.token_id or not credentials.token_key:
            raise ValueError(
                "VNPT embedding credentials not configured. "
                "Please set VNPT_CONFIG_FILE or VNPT_API_KEY_EMBEDDING environment variables."
            )
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session is initialized"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get VNPT API authentication headers per API documentation"""
        credentials = self.config.get_credentials("embedding")
        if not credentials:
            raise ValueError("No embedding credentials found in VNPT config")
        
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Token-id": credentials.token_id,
            "Token-key": credentials.token_key,
            "Content-Type": "application/json",
        }
    
    async def embed(
        self,
        text: str,
        encoding_format: str = "float"
    ) -> List[float]:
        """
        Get embedding for single text
        
        Args:
            text: Text to embed (supports Vietnamese)
            encoding_format: Format of embedding ("float" or "base64")
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = await self.embed_batch([text], encoding_format=encoding_format)
        return embeddings[0] if embeddings else []
    
    async def embed_batch(
        self,
        texts: List[str],
        encoding_format: str = "float",
        max_concurrent: Optional[int] = None
    ) -> List[List[float]]:
        """
        Get embeddings for batch of texts with concurrency control
        
        Args:
            texts: List of texts to embed (supports Vietnamese)
            encoding_format: Format of embedding ("float" or "base64")
            max_concurrent: Maximum concurrent requests (default: 8, respects quota)
            
        Returns:
            List of embedding vectors
            
        Raises:
            RuntimeError: If API calls fail
        """
        if max_concurrent is None:
            max_concurrent = self.max_concurrent
        
        session = await self._ensure_session()
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def embed_one(text: str) -> List[float]:
            """Embed single text with semaphore control"""
            async with semaphore:
                # Build request payload per VNPT API spec (Section 3.3)
                payload = {
                    "model": self.MODEL_NAME,
                    "input": text,
                    "encoding_format": encoding_format,
                }
                
                headers = self._get_auth_headers()
                endpoint = self.API_BASE_URL + self.EMBEDDING_ENDPOINT
                
                try:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
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
                        f"Embedding API request timeout after {self.timeout}s"
                    )
        
        # Execute concurrent requests with semaphore
        embeddings = await asyncio.gather(
            *[embed_one(text) for text in texts],
            return_exceptions=False
        )
        
        return embeddings
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()

