from abc import ABC, abstractmethod
from optparse import Option
from token import OP
from typing import Any, Dict, List, Optional

Tool_Set = Dict[str, Any]

class LLMServiceConfig(ABC):
    provider: str
    model: str

class LLMService(ABC):
    '''
    The LLMService interface provides a contract for interacting with an LLM service.
    It defines the methods for generating text, retrieving available tools, and retrieving the service configuration.
    '''
    @abstractmethod
    async def generate(
        self,
        user_input: str,
        stream: Optional[bool] = False,
    ) -> str:
        pass

    @abstractmethod
    def get_all_tools(
        self    
    ) -> Tool_Set:
        pass

    @abstractmethod
    def get_config(
        self
    ) -> LLMServiceConfig:
        pass
