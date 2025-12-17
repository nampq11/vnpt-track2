from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict
from src.brain.agent.prompts import READING_PROMPT
import re
import json

class ReadingTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        # Use provided LLM service or fallback to VNPT large model
        if isinstance(llm_service, VNPTService):
            # If VNPT, use large model for better comprehension
            self.llm_service = VNPTService(
                model="vnptai-hackathon-large",
                model_type="large",
            )
            logger.info("Initialized Reading Task with VNPT large model")
        else:
            # Use provided service (Azure, Ollama, etc.)
            self.llm_service = llm_service
            logger.info(f"Initialized Reading Task with {llm_service.__class__.__name__}")

    def _extract_question_from_context(self, query: str) -> tuple:
        """Separate context from question in reading comprehension queries."""
        # Common patterns for question markers
        markers = [
            "Câu hỏi:",
            "Hỏi:",
            "Question:",
            "\nCâu hỏi ",
        ]
        
        for marker in markers:
            if marker in query:
                parts = query.rsplit(marker, 1)
                if len(parts) == 2:
                    return parts[0].strip(), marker + parts[1].strip()
        
        # If no marker found, assume last sentence is question
        # Find last sentence that ends with ?
        sentences = query.split('?')
        if len(sentences) > 1:
            question = sentences[-2].split('.')[-1] + '?'
            context = query.replace(question, '').strip()
            return context, question
        
        # Fallback: return as-is (entire query is both context and question)
        return query, query

    def _format_choices(self, options: Dict[str, str]) -> str:
        """Format choices for prompt."""
        return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    def _parse_json_answer(self, text: str, options: Dict[str, str]) -> Dict[str, str]:
        """Extract JSON answer from LLM response with CoT reasoning."""
        try:
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if "answer" in data:
                    answer = data["answer"].upper().strip()
                    if answer in options:
                        result = {"answer": answer}
                        # Log reasoning if present (for debugging/analysis)
                        if "reasoning" in data:
                            logger.debug(f"CoT Reasoning: {data['reasoning'][:200]}...")
                        return result
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
    ) -> Dict[str, str]:
        try:
            context, question = self._extract_question_from_context(query)
            choices_str = self._format_choices(options)
            
            prompt = READING_PROMPT.format(
                context=context,
                question=question,
                choices=choices_str,
            )

            logger.info(
                f"Reading Task Prompt: {prompt}"
            )
            
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            logger.debug(f"Reading Task LLM response: {response_text}")
            
            return self._parse_json_answer(response_text, options)
            
        except Exception as e:
            logger.error(f"Error invoking Reading Task: {e}")
            return {"answer": "A"}