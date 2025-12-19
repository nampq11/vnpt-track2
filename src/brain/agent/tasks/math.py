from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from src.models.tasks.math import DomainMathTask
from src.brain.utils.json_parser import extract_answer_from_response
from src.brain.system_prompt import EnhancedPromptManager, PromptType


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
        
        self.prompt_manager = EnhancedPromptManager.get_instance()

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
        query_id: str = None,
        verbose: bool = False,
    ) -> Dict[str, str]:
        try:
            if verbose:
                logger.debug(f"[{query_id}] Math Task invoked with query: {query[:50]}... and {len(options)} options and domain: {domain}")
            choices_str = self._format_choices(options)

            if domain == DomainMathTask.CHEMISTRY:
                system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.CHEMISTRY)
            else:
                system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.MATH)
            
            user_prompt = user_template.format(
                    query=query,
                    choices=choices_str,
                )
            
            if verbose:
                logger.info(f"[{query_id}] Math Task calling LLM with user prompt: {user_prompt}")
            response_text = await self.llm_service.generate(
                user_input=user_prompt,
                system_message=system_prompt,
            )
            if verbose:
                logger.debug(f"[{query_id}] Math Task LLM response: {response_text}")
            result = self._parse_json_answer(response_text, options)
            if verbose:
                logger.debug(f"[{query_id}] Math Task parsed result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error invoking Math Task: {e}")
            return {"answer": "A"}
