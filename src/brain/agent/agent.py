import aiohttp
from src.brain.agent.tasks.rag import RAGTask
from src.brain.llm.services.type import LLMService
from src.brain.agent.query_classification import QueryClassificationService
from src.brain.agent.guardrail import GuardrailService
from loguru import logger
from typing import Any, Dict
import time

from src.models.agent import ScenarioTask
from src.brain.agent.tasks.math import MathTask
from src.brain.agent.tasks.reading import ReadingTask

class Agent:
    def __init__(
        self,
        llm_service: LLMService,
    ) -> None:
        self.llm_service = llm_service
        self.query_classification = QueryClassificationService(llm_service=llm_service)
        self.guardrail = GuardrailService(llm_service=llm_service)
        
        # --- TASKS ---
        self.math_task = MathTask(llm_service=llm_service)
        self.reading_task = ReadingTask(llm_service=llm_service)
        self.rag_task = RAGTask(llm_service=llm_service)
        logger.info("Initialized Agent")

    async def process_query(
        self,
        query: str,
        options: Dict[str, str],
        query_id: str,
    ) -> Dict[str, Any]:
        start_time = time.time()

        try:
            # --- LAYER 1: FAST SAFE CHECK ---
            try:
                async with aiohttp.ClientSession() as session:
                    embedding_response = await self.llm_service.get_embedding(
                        session=session,
                        text=query,
                    )
                    # Extract embedding vector from response
                    # VNPT API returns dict with 'data' containing embeddings
                    if isinstance(embedding_response, dict) and 'data' in embedding_response:
                        query_embedding = embedding_response['data'][0]['embedding']
                    else:
                        query_embedding = embedding_response
                    
                    guardrail_result = await self.guardrail.invoke(
                        user_input=query,
                        embedding=query_embedding,
                    )
                    logger.info(f"Guardrail Result: {guardrail_result}")
            except Exception as e:
                logger.warning(f"Guardrail check failed (non-blocking): {e}")
                guardrail_result = (True, {})
            # --- LAYER 2: QUERY CLASSIFICATION ---
            classification = await self.query_classification.invoke(
                query=query,
            )
            logger.info(f"Query Classification Result: {classification}")

            # --- LAYER 3: EXECUTION ---
            if classification['category'] == ScenarioTask.SAFETY:
                result = await self.guardrail.invoke(
                    user_input=query,
                    is_safe=False,
                    options=options,
                )
                return result
            elif classification['category'] == ScenarioTask.MATH:
                result = await self.math_task.invoke(
                    query=query,
                    options=options,
                )
            elif classification['category'] == ScenarioTask.READING:
                result = await self.reading_task.invoke(
                    query=query,
                    options=options,
                )
            elif classification['category'] == ScenarioTask.RAG:
                result = await self.rag_task.invoke(
                    query=query,
                    options=options,
                    temporal_constraint=classification['temporal_constraint'],
                    key_entities=classification['key_entities'],
                )
            else:
                result = {
                    "answer": "A",
                }
            return result
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise e