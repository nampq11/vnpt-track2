from src.brain.llm.services.type import LLMService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from typing import List

class RAGTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        super().__init__(llm_service=llm_service)
        logger.info("Initialized RAG Task")

    async def invoke(
        self,
        query: str,
        options: Dict[str, str],
        temporal_constraint: int,
        key_entities: List[str],
    ) -> str:
        return "RAG Task"