from src.brain.llm.services.type import LLMService
from loguru import logger
from typing import Any, Dict
from src.brain.utils.json_parser import parse_json_from_llm_response
from src.brain.system_prompt import EnhancedPromptManager, PromptType


class QueryClassificationService:
    def __init__(
        self,
        llm_service: LLMService
    ) -> None:
        self.llm_service = llm_service
        self.prompt_manager = EnhancedPromptManager.get_instance()
        logger.info("Initialized Query Classification Service")

    async def invoke(
        self,
        query: str
    ) -> Dict[str, Any]:
        result = None
        try:
            system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.CLASSIFICATION)
            
            user_prompt = user_template.format(query=query)
            response_text = await self.llm_service.generate(
                user_input=user_prompt,
                system_message=system_prompt
            )
            result = self._parse_json_answer_robust(response_text)
            logger.info(f"Query Classification Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error invoking Query Classification Service: {e}")
            return {
                "category": "RAG",
                "temporal_constraint": None,
                "reasoning": "Error invoking Query Classification Service",
                "key_entities": []
            }

    def _parse_json_answer_robust(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback to default classification"""
        logger.info(f"Parsing JSON answer: {text[:100]}...")
        return parse_json_from_llm_response(
            text,
            default={
                "category": "RAG",
                "temporal_constraint": None,
                "reasoning": "JSON Parsing Error",
                "key_entities": []
            },
            context="QueryClassification"
        )
