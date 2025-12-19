from brain.llm.services.vnpt import VNPTService
from src.brain.llm.services.type import LLMService
from loguru import logger
from typing import Any, Dict
from src.brain.utils.json_parser import parse_json_from_llm_response
from src.brain.system_prompt import EnhancedPromptManager, PromptType
from src.models.tasks.rag import DomainRAGTask
from src.models.tasks.math import DomainMathTask
from src.models.agent import ScenarioTask


class QueryClassificationService:
    def __init__(
        self,
        llm_service: LLMService,
        verbose: bool = False
    ) -> None:
        if isinstance(llm_service, VNPTService):
            # If VNPT, use large model for better reasoning
            self.llm_service = VNPTService(
                model="vnptai-hackathon-large",
                model_type="large",
            )
            logger.info("Initialized Query Classification Service with VNPT large model")
        else:
            # Use provided service (Azure, Ollama, etc.)
            self.llm_service = llm_service
            logger.info(f"Initialized Query Classification Service with {llm_service.__class__.__name__}")
        self.prompt_manager = EnhancedPromptManager.get_instance()
        if verbose:
            logger.info("Initialized Query Classification Service")

    async def invoke(
        self,
        query: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        result = None
        try:
            system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.CLASSIFICATION)
            
            user_prompt = user_template.format(query=query)
            response_text = await self.llm_service.generate(
                user_input=user_prompt,
                system_message=system_prompt
            )
            result = self._parse_json_answer_robust(response_text, verbose=verbose)
            if verbose:
                logger.info(f"Query Classification Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error invoking Query Classification Service: {e}")
            return {
                "category": ScenarioTask.RAG,
                "domain": DomainRAGTask.GENERAL_KNOWLEDGE,
                "temporal_constraint": None,
                "reasoning": "Error invoking Query Classification Service",
                "key_entities": []
            }

    def _parse_json_answer_robust(
        self,
        text: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Parse JSON from LLM response with fallback to default classification"""
        if verbose:
            logger.info(f"Parsing JSON answer: {text[:100]}...")
        default_classification = parse_json_from_llm_response(
            text,
            default={
                "category": ScenarioTask.RAG,
                "domain": DomainRAGTask.GENERAL_KNOWLEDGE,
                "temporal_constraint": None,
                "reasoning": "JSON Parsing Error",
                "key_entities": []
            },
            context="QueryClassification"
        )
        
        # Ensure domain is set based on category if missing
        if "domain" not in default_classification:
            category = default_classification.get("category", ScenarioTask.RAG)
            if category == ScenarioTask.MATH:
                default_classification["domain"] = DomainMathTask.MATH
            elif category == ScenarioTask.RAG:
                default_classification["domain"] = DomainRAGTask.GENERAL_KNOWLEDGE
        
        return default_classification
