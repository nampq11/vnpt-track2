from src.brain.llm.services.type import LLMService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from src.brain.agent.prompts import MATH_PROMPT

class MathTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        super().__init__(llm_service=llm_service)
        logger.info("Initialized Math Task")

    async def invoke(
        self,
        query: str,
        options: Dict[str, str],
    ) -> str:
        try:
            prompt = MATH_PROMPT.format(
                query=query,
                choices=options,
            )
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            return response_text
        except Exception as e:
            logger.error(f"Error invoking Math Task: {e}")
            raise e