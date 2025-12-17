from src.brain.llm.services.type import LLMService
from loguru import logger
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from src.brain.agent.prompts import SAFETY_SELECTOR_PROMPT
import re
from dotenv import load_dotenv
import json
import os

load_dotenv()

class GuardrailService:
    def __init__(
        self,
        llm_service: LLMService
    ) -> None:
        self.llm_service = llm_service
        
        # Get the safety index path from environment or use default
        safety_index_path = os.getenv('SAFETY_INDEX_PATH', 'data/embeddings/safety_index.npy')
        
        # Convert to absolute path if relative
        if not os.path.isabs(safety_index_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            safety_index_path = os.path.join(project_root, safety_index_path)
        
        if not os.path.exists(safety_index_path):
            raise FileNotFoundError(
                f"Safety index not found at {safety_index_path}. "
                f"Please set SAFETY_INDEX_PATH environment variable or ensure the file exists."
            )
        
        self.safety_index = np.load(safety_index_path)
        self.safety_threshold = 0.9

        # Load safety queries
        safety_queries_path = os.getenv('SAFETY_QUERIES_PATH', 'data/embeddings/safety_queries.json')
        if not os.path.isabs(safety_queries_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            safety_queries_path = os.path.join(project_root, safety_queries_path)
        
        if not os.path.exists(safety_queries_path):
            raise FileNotFoundError(
                f"Safety queries not found at {safety_queries_path}. "
                f"Please set SAFETY_QUERIES_PATH environment variable or ensure the file exists."
            )
        
        with open(safety_queries_path, 'r', encoding='utf-8') as f:
            self.safety_queries = json.load(f)
        
        logger.info(f"Loaded {len(self.safety_queries)} safety queries from {safety_queries_path}")

    async def invoke(
        self,
        user_input: str,
        is_safe: Optional[bool] = None,
        embedding: Optional[List[float]] = None,
        options: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, Dict[str, str]]:
        try:
            violation_reason = None
            matched_query = None

            # Check embedding safety if available
            if embedding is not None:
                # Convert embedding to numpy array of floats
                embedding_array = np.array(embedding, dtype='float32')
                
                # Validate dimensions
                if embedding_array.shape[0] != self.safety_index.shape[1]:
                    logger.warning(f"Embedding dimension mismatch: {embedding_array.shape[0]} != {self.safety_index.shape[1]}")
                    is_safe = True
                else:
                    # Normalize embedding for proper cosine similarity
                    norm = np.linalg.norm(embedding_array)
                    if norm > 0:
                        embedding_array = embedding_array / norm
                    
                    scores = np.dot(self.safety_index, embedding_array)
                    max_score = float(np.max(scores))
                    violation_reason = f"Similarity {max_score:.2f}"
                    logger.info(f"violation reason: {violation_reason}")
                    
                    if max_score > self.safety_threshold:
                        max_score_idx = int(np.argmax(scores))
                        matched_query = self.safety_queries[max_score_idx]
                        logger.info(f"matched safety query: {matched_query}")
                        is_safe = False
                    else:
                        is_safe = True

            # If violation detected, call LLM to get safe answer
            if user_input is not None and not is_safe:
                prompt = SAFETY_SELECTOR_PROMPT.format(
                    query=user_input,
                    violation_reason=violation_reason or "None",
                    options=options
                )
                try:
                    response_text = await self.llm_service.generate(
                        user_input=prompt,
                    )

                    result = self._parse_json_answer_robust(options, response_text)
                    logger.info(f"Guardrail Service Result: {result}")
                    return is_safe, result
                except Exception as e:
                    logger.error(f"Error invoking Guardrail Service: {e}")
                    return is_safe, {}
            
            # No violation detected, return empty dict
            return is_safe, {}
        except Exception as e:
            logger.error(f"Error invoking Guardrail Service: {e}")
            raise e

    def _parse_json_answer_robust(
        self,
        options: Optional[Dict[str, str]],
        text: str
    ) -> Dict[str, str]:
        """Parse JSON answer from LLM response and return as dict"""
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            logger.info(f"Parsing JSON answer: {text} with options: {options}")
            
            if match:
                data = json.loads(match.group())
                answer = data.get("answer", "None")
                return {"answer": answer}
        except Exception as e:
            logger.error(f"Error parsing JSON answer: {e}")
        
        # Fallback: return first option if available
        if options:
            return {"answer": sorted(options.keys())[0]}
        return {"answer": "A"}