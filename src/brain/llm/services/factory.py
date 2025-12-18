from typing import Literal, Optional
from src.brain.llm.services.type import LLMService
from src.brain.llm.services.ollama import OllamaService
from src.brain.llm.services.vnpt import VNPTService
from src.brain.llm.services.azure import AzureService

class LLMFactory:
    """Factory for creating LLM services"""
    
    @staticmethod
    def create(
        provider: Literal["ollama", "vnpt", "azure"],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMService:
        """
        Create an LLM service instance
        
        Args:
            provider: Service provider name
            model: Model name (optional)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized LLMService
        """
        if provider == "vnpt":
            model_name = model or "vnptai-hackathon-small"
            model_type = "small" if "small" in model_name else "large"
            return VNPTService(
                model=model_name,
                model_type=model_type,
                **kwargs
            )
            
        elif provider == "azure":
            return AzureService(
                model=model or "gpt-4.1",
                **kwargs
            )
            
        elif provider == "ollama":
            return OllamaService(
                model=model or "qwen3:1.7b",
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown provider: {provider}")

