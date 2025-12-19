"""Prompt Registry for managing typed prompt access."""
from enum import Enum
from typing import Dict, Optional, Tuple


class PromptType(Enum):
    """Available prompt types in the system."""
    CLASSIFICATION = "classification"
    SAFETY = "safety"
    MATH = "math"
    CHEMISTRY = "chemistry"
    READING = "reading"
    RAG = "rag"
    RAG_WITH_CONTEXT = "rag_with_context"


class PromptRegistry:
    """Registry for storing and retrieving prompt templates."""
    
    _instance: Optional["PromptRegistry"] = None
    
    def __init__(self) -> None:
        self._prompts: Dict[str, Dict[str, str]] = {}
    
    @classmethod
    def get_instance(cls) -> "PromptRegistry":
        """Get singleton instance of PromptRegistry."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
    
    def register(self, prompt_type: str, system: str, user: str) -> None:
        """Register a prompt pair (system + user template)."""
        self._prompts[prompt_type] = {
            "system": system,
            "user": user
        }
    
    def get_system(self, prompt_type: PromptType) -> str:
        """Get system prompt for a given type."""
        key = prompt_type.value
        if key not in self._prompts:
            raise KeyError(f"Prompt type '{key}' not registered")
        return self._prompts[key]["system"]
    
    def get_user(self, prompt_type: PromptType) -> str:
        """Get user prompt template for a given type."""
        key = prompt_type.value
        if key not in self._prompts:
            raise KeyError(f"Prompt type '{key}' not registered")
        return self._prompts[key]["user"]
    
    def get_prompts(self, prompt_type: PromptType) -> Tuple[str, str]:
        """Get both system and user prompts for a given type."""
        return self.get_system(prompt_type), self.get_user(prompt_type)
    
    def is_registered(self, prompt_type: PromptType) -> bool:
        """Check if a prompt type is registered."""
        return prompt_type.value in self._prompts
    
    @property
    def registered_types(self) -> list:
        """Get list of registered prompt types."""
        return list(self._prompts.keys())

