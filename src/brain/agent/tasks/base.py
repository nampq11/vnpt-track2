from abc import ABC, abstractmethod
from src.brain.llm.services.type import LLMService
from loguru import logger
from typing import Dict

class BaseTask(ABC):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        self.llm_service = llm_service
        logger.info("Initialized Base Task")

    @abstractmethod
    async def invoke(
        self,
        query: str,
        options: Dict[str, str],
    ) -> str:
        pass