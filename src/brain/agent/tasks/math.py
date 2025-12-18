from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from src.brain.agent.prompts import CHEMISTRY_PROMPT, MATH_PROMPT
from src.models.tasks.math import DomainMathTask
from src.brain.utils.json_parser import extract_answer_from_response

class MathTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        # Use provided LLM service or fallback to VNPT large model
        if isinstance(llm_service, VNPTService):
            # If VNPT, use large model for better reasoning
            self.llm_service = VNPTService(
                model="vnptai-hackathon-large",
                model_type="large",
            )
            logger.info("Initialized Math Task with VNPT large model")
        else:
            # Use provided service (Azure, Ollama, etc.)
            self.llm_service = llm_service
            logger.info(f"Initialized Math Task with {llm_service.__class__.__name__}")

    def _format_choices(self, options: Dict[str, str]) -> str:
        """Format choices for prompt."""
        return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    def _parse_json_answer(self, text: str, options: Dict[str, str]) -> Dict[str, str]:
        """Extract JSON answer from LLM response."""
        return extract_answer_from_response(text, options)

    async def invoke(
        self,
        query: str,
        domain: DomainMathTask,
        options: Dict[str, str],
    ) -> Dict[str, str]:
        try:
            logger.debug(f"Math Task invoked with query: {query[:50]}... and {len(options)} options and domain: {domain}")
            choices_str = self._format_choices(options)
            prompt = ""
            if domain == DomainMathTask.MATH:
                prompt = MATH_PROMPT.format(
                    query=query,
                    choices=choices_str,
                )
            elif domain == DomainMathTask.CHEMISTRY:
                prompt = CHEMISTRY_PROMPT.format(
                    query=query,
                    choices=choices_str,
                )
            else:
                prompt = MATH_PROMPT.format(
                    query=query,
                    choices=choices_str,
                )
            logger.info(f"Math Task calling LLM with prompt: {prompt}")
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            logger.debug(f"Math Task LLM response: {response_text}")
            result = self._parse_json_answer(response_text, options)
            logger.debug(f"Math Task parsed result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error invoking Math Task: {e}")
            return {"answer": "A"}