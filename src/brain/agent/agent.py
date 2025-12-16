from src.brain.llm.services.type import LLMService
from src.brain.agent.query_classification import QueryClassificationService
from src.brain.agent.guardrail import GuardrailService
from loguru import logger
from typing import Any, Dict
import time

class Agent:
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        self.llm_service = llm_service
        self.query_classification = QueryClassificationService(llm_service=llm_service)
        self.guardrail = GuardrailService(llm_service=llm_service)

        logger.info("Initialized Agent")

    async def process_query(
        self,
        query: str,
        options: Dict[str, str],
        query_id: str,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # --- LAYER 1: FAST SAFE CHECK ---
            query_embedding = await self.llm_service.get_embedding(
                session=
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise e