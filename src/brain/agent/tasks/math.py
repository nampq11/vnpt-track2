from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from src.brain.agent.prompts import MATH_PROMPT
import re
import json

class MathTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        # Create dedicated large model service for math reasoning
        self.llm_service = VNPTService(
            model="vnptai-hackathon-large",
            model_type="large",
        )
        logger.info("Initialized Math Task with large model")

    def _format_choices(self, options: Dict[str, str]) -> str:
        """Format choices for prompt."""
        return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    def _parse_json_answer(self, text: str, options: Dict[str, str]) -> Dict[str, str]:
        """Extract JSON answer from LLM response."""
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "answer" in data:
                    answer = data["answer"].upper().strip()
                    if answer in options:
                        return {"answer": answer}
        except Exception as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        # Fallback: find first valid option letter
        text_upper = text.upper()
        for letter in ["A", "B", "C", "D"]:
            if letter in options and f'"{letter}"' in text_upper:
                return {"answer": letter}
        
        return {"answer": "A"}

    async def invoke(
        self,
        query: str,
        options: Dict[str, str],
    ) -> Dict[str, str]:
        try:
            choices_str = self._format_choices(options)
            prompt = MATH_PROMPT.format(
                query=query,
                choices=choices_str,
            )
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            logger.debug(f"Math Task LLM response: {response_text}")
            return self._parse_json_answer(response_text, options)
        except Exception as e:
            logger.error(f"Error invoking Math Task: {e}")
            return {"answer": "A"}