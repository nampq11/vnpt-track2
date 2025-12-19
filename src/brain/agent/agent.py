import aiohttp
import asyncio
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
        verbose: bool = False
    ) -> None:
        self.llm_service = llm_service
        self.verbose = verbose
        self.query_classification = QueryClassificationService(llm_service=llm_service)
        self.guardrail = GuardrailService(llm_service=llm_service)
        
        # --- TASKS ---
        self.math_task = MathTask(llm_service=llm_service)
        self.reading_task = ReadingTask(llm_service=llm_service)
        self.rag_task = RAGTask(llm_service=llm_service)
        if verbose:
            logger.info("Initialized Agent")

    async def process_query(
        self,
        query: str,
        options: Dict[str, str],
        query_id: str,
        timeout: float = 60.0,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Process query with timeout protection.
        
        Args:
            query: User query text
            options: Answer options (A/B/C/D)
            query_id: Unique query identifier
            timeout: Max processing time in seconds (default: 60)
        
        Returns:
            Dict with "answer" key
        """
        try:
            return await asyncio.wait_for(
                self._process_query_internal(query, options, query_id, verbose),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"[{query_id}] Query timed out after {timeout}s")
            # Return safe fallback answer
            fallback = sorted(options.keys())[0] if options else "A"
            return {"answer": fallback}
        except Exception as e:
            logger.error(f"[{query_id}] Error processing query: {e}")
            raise e
    
    async def _process_query_internal(
        self,
        query: str,
        options: Dict[str, str],
        query_id: str,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Internal query processing logic with 3-layer architecture"""
        start_time = time.time()

        try:
            if verbose:
                logger.info(f"[{query_id}] Processing query: {query[:50]}...")
            # --- LAYER 1: FAST SAFE CHECK ---
            try:
                async with aiohttp.ClientSession() as session:
                    query_embedding = await self.llm_service.get_embedding(
                        session=session,
                        text=query,
                    )
                    
                    if query_embedding:
                        guardrail_result = await self.guardrail.invoke(
                            user_input=query,
                            embedding=query_embedding,
                            options=options,  # Pass options for answer generation
                        )
                        if verbose: 
                            logger.info(f"[{query_id}] Guardrail Result: {guardrail_result}")
                    else:
                        guardrail_result = (True, {})
            except Exception as e:
                logger.warning(f"Guardrail check failed (non-blocking): {e}")
                guardrail_result = (True, {})
            
            # Check if guardrail detected unsafe content
            is_safe, guardrail_answer = guardrail_result
            
            # If guardrail blocked the query, return the safe answer immediately
            if not is_safe:
                if verbose:
                   logger.warning(f"[{query_id}] Query blocked by guardrail embedding check")
                # Validate answer is not 'None'
                if guardrail_answer.get('answer') not in [None, 'None', '']:
                    return guardrail_answer
                else:
                    # Fallback to first option if guardrail couldn't determine answer
                    if verbose:
                        logger.warning(f"[{query_id}] Guardrail returned invalid answer, using fallback")
                    return {"answer": sorted(options.keys())[0]}
            
            # --- LAYER 2: QUERY CLASSIFICATION ---
            classification = await self.query_classification.invoke(
                query=query,
            )
            if verbose:
                logger.info(f"[{query_id}] Query Classification Result: {classification}")

            # --- LAYER 3: EXECUTION ---
            if classification['category'] == ScenarioTask.SAFETY:
                # Classification identified as safety (separate from guardrail)
                # Call guardrail with options to get appropriate response
                is_safe_check, safe_answer = await self.guardrail.invoke(
                    user_input=query,
                    is_safe=False,
                    options=options,
                    verbose=verbose,
                )
                result = safe_answer if isinstance(safe_answer, dict) else {"answer": sorted(options.keys())[0]}
            elif classification['category'] == ScenarioTask.MATH:
                # Safe access to domain with fallback
                domain = classification.get('domain', None)
                if not domain:
                    logger.warning(f"[{query_id}] Missing domain in classification, using default MATH domain")
                    from src.models.tasks.math import DomainMathTask
                    domain = DomainMathTask.MATH
                result = await self.math_task.invoke(
                    query=query,
                    domain=domain,
                    options=options,
                    verbose=verbose,
                )
            elif classification['category'] == ScenarioTask.READING:
                result = await self.reading_task.invoke(
                    query=query,
                    options=options,
                    verbose=verbose,
                )
            elif classification['category'] == ScenarioTask.RAG:
                # Safe access to domain with fallback
                domain = classification.get('domain', None)
                if not domain:
                    logger.warning(f"[{query_id}] Missing domain in classification, using default GENERAL_KNOWLEDGE domain")
                    from src.models.tasks.rag import DomainRAGTask
                    domain = DomainRAGTask.GENERAL_KNOWLEDGE
                result = await self.rag_task.invoke(
                    query=query,
                    domain=domain,
                    options=options,
                    temporal_constraint=classification.get('temporal_constraint'),
                    key_entities=classification.get('key_entities', []),
                    verbose=verbose,
                )
            else:
                result = {
                    "answer": "A",
                }
            
            logger.debug(f"[{query_id}] Task execution result: {result}")
            return result
                
        except Exception as e:
            logger.error(f"[{query_id}] Error in internal processing: {e}")
            raise e