from src.brain.llm.services.type import LLMService
from loguru import logger
from typing import Any, Dict
from src.brain.agent.prompts import QUERY_CLASSIFICATION_PROMPT
import re
import json

class QueryClassificationService:
    def __init__(
        self,
        llm_service: LLMService
    ) -> None:
        self.llm_service = llm_service
        logger.info("Initialized Query Classification Service")

    async def invoke(
        self,
        query: str
    ) -> Dict[str, Any]:
        result = None
        try:
            prompt = QUERY_CLASSIFICATION_PROMPT.format(
                query=query
            )
            response_text = await self.llm_service.generate(
                user_input=prompt
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
        try:
            logger.info(f"Parsing JSON answer: {text}")
            match = re.search(r'\{.*\}', text, re.DOTALL)
            logger.info(f"Match: {match}")
            if match:
                data = json.loads(match.group())
                return data
        except Exception as e:
            logger.error(f"Error parsing JSON answer: {e}")
            return {
                "category": "RAG",
                "temporal_constraint": None,
                "reasoning": "JSON Parsing Error",
                "key_entities": []
            }