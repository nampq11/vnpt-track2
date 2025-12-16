from brain.llm.services.type import LLMService
from loguru import logger
import numpy as np
from typing import Any, Dict, List, Tuple
from brain.agent.prompts import SAFETY_SELECTOR_PROMPT
import re
import json

class GuardrailService:
    def __init__(
        self,
        llm_service: LLMService
    ) -> None:
        self.llm_service = llm_service
        self.safety_index = np.load('./data/embeddings/safety_index.npy')
        self.safety_threshold = 0.85
        logger.info("Initialized Guardrail Service")

    async def invoke(
        self,
        embedding: List[float],
        user_input: str,
        options: Dict[str, str],
        query_id: str
    ) -> Tuple[bool, str]:
        try:
            if embedding is not None:
                scores = np.dot(self.safety_index, embedding)
                if np.max(scores) > self.safety_threshold:
                    return False, f"Similarity {np.max(scores):.2f}"
            
                violation_reason = f"Similarity {np.max(scores):.2f}"

            if user_input is not None:
                prompt = SAFETY_SELECTOR_PROMPT.format(
                    query=user_input,
                    violation_reason=violation_reason,
                    options=options
                )
                try:
                    response_text = await self.llm_service.generate(
                        user_input=prompt,
                    )

                    return self._parse_json_answer_robust(response_text)
                except Exception as e:
                    logger.error(f"Error invoking Guardrail Service: {e}")
                    return False, "Error parsing JSON answer"
        except Exception as e:
            logger.error(f"Error invoking Guardrail Service: {e}")
            raise e

    def _parse_json_answer_robust(
        self,
        options: Dict[str, str],
        text: str
    ) -> Dict[str, Any]:
        try:
            match = re.search(r'\{.*\}', text)
            if match:
                data = json.loads(match.group())
                if "answer" in data and data["answer"] in options["choices"]:
                    return {"answer": data["answer"]}
        except Exception as e:
            logger.error(f"Error parsing JSON answer: {e}")
            return None