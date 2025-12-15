"""Abstract LLM service interface for runtime inference"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class LLMServiceConfig:
    """Configuration for LLM service"""
    provider: str
    model: str


@dataclass
class LLMServiceResponse:
    """Response from LLM service"""
    content: str
    stop_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None


class LLMService(ABC):
    """Abstract base class for LLM service providers"""
    
    @abstractmethod
    async def generate(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMServiceResponse:
        """
        Generate response from LLM
        
        Args:
            user_input: User query/prompt
            system_prompt: System message to guide behavior
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLMServiceResponse with generated content
        """
        pass
    
    @abstractmethod
    async def embed(
        self,
        text: str,
    ) -> List[float]:
        """
        Get embedding vector for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        pass
    
    @abstractmethod
    def get_config(self) -> LLMServiceConfig:
        """Get service configuration"""
        pass

