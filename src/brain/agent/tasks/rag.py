from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict, List, Optional
from src.brain.agent.prompts import RAG_PROMPT
import re
import json

class RAGTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        # Use provided LLM service or fallback to VNPT large model
        if isinstance(llm_service, VNPTService):
            # If VNPT, use large model for better knowledge retrieval
            self.llm_service = VNPTService(
                model="vnptai-hackathon-large",
                model_type="large",
            )
            logger.info("Initialized RAG Task with VNPT large model")
        else:
            # Use provided service (Azure, Ollama, etc.)
            self.llm_service = llm_service
            logger.info(f"Initialized RAG Task with {llm_service.__class__.__name__}")

    def _format_choices(self, options: Dict[str, str]) -> str:
        """Format choices for prompt."""
        return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    def _build_temporal_hint(self, temporal_constraint: Optional[int]) -> str:
        """Build temporal context hint."""
        if temporal_constraint:
            return f"Note: This question relates to year {temporal_constraint}.\n"
        return ""

    def _build_entities_hint(self, key_entities: Optional[List[str]]) -> str:
        """Build key entities hint."""
        if key_entities and len(key_entities) > 0:
            entities_str = ", ".join(key_entities)
            return f"Key concepts: {entities_str}\n"
        return ""

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
        for letter in sorted(options.keys()):
            if f'"{letter}"' in text_upper or f"'{letter}'" in text_upper:
                return {"answer": letter}
        
        # Last resort: return first option
        return {"answer": sorted(options.keys())[0]}

    async def invoke(
        self,
        query: str,
        options: Dict[str, str],
        temporal_constraint: Optional[int] = None,
        key_entities: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        try:
            choices_str = self._format_choices(options)
            temporal_hint = self._build_temporal_hint(temporal_constraint)
            entities_hint = self._build_entities_hint(key_entities)
            
            prompt = RAG_PROMPT.format(
                query=query,
                temporal_hint=temporal_hint,
                entities_hint=entities_hint,
                choices=choices_str,
            )
            
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            logger.debug(f"RAG Task LLM response: {response_text}")
            
            return self._parse_json_answer(response_text, options)
            
        except Exception as e:
            logger.error(f"Error invoking RAG Task: {e}")
            return {"answer": "A"}