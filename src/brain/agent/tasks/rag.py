from src.models.tasks.rag import DomainRAGTask
from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict, List, Optional
from src.brain.agent.domain_mapper import DomainMapper
from src.brain.utils.json_parser import extract_answer_from_response
from src.brain.system_prompt import EnhancedPromptManager, PromptType
import os


class RAGTask(BaseTask):
    def __init__(
        self,
        llm_service: LLMService,
        retriever: Optional[any] = None,
        use_retrieval: bool = True,
        retrieval_top_k: int = 3,
        index_dir: str = "data/embeddings/knowledge",
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
        
        self.use_retrieval = use_retrieval
        self.retrieval_top_k = retrieval_top_k
        self.domain_mapper = DomainMapper()
        self.prompt_manager = EnhancedPromptManager.get_instance()
        
        # Initialize retriever if enabled and index exists
        if use_retrieval and retriever is not None:
            self.retriever = retriever
        elif use_retrieval and os.path.exists(index_dir):
            try:
                from src.brain.rag.lancedb_retriever import LanceDBRetriever
                self.retriever = LanceDBRetriever.from_directory(
                    index_dir=index_dir,
                    llm_service=llm_service,
                )
                logger.info(f"Loaded LanceDB retriever from {index_dir}")
            except Exception as e:
                logger.warning(f"Failed to load LanceDB retriever: {e}")
                self.retriever = None
        else:
            self.retriever = None
            if use_retrieval:
                logger.warning(f"Retrieval enabled but index not found at {index_dir}")

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

    def _parse_json_answer(self, text: str, options: Dict[str, str], verbose: bool = False) -> Dict[str, str]:
        """Extract JSON answer from LLM response with CoT reasoning."""
        return extract_answer_from_response(text, options, verbose=verbose)

    async def invoke(
        self,
        query: str,
        domain: DomainRAGTask,
        options: Dict[str, str],
        temporal_constraint: Optional[int] = None,
        key_entities: Optional[List[str]] = None,
        query_id: str = None,
        verbose: bool = False,
    ) -> Dict[str, str]:
        try:
            if verbose:
                logger.info(f"[{query_id}] RAG Task invoked with domain={domain}, query={query[:50]}...")
            choices_str = self._format_choices(options)
            temporal_hint = self._build_temporal_hint(temporal_constraint)
            entities_hint = self._build_entities_hint(key_entities)
            
            # Get domain-specific retrieval configuration
            retrieval_config = self.domain_mapper.get_retrieval_config(domain)
            
            # Attempt retrieval if enabled
            context = ""
            if self.retriever is not None:
                try:
                    from src.brain.rag.lancedb_retriever import format_retrieval_context
                    
                    # --- DOMAIN-AWARE FILTERING ---
                    # Use domain mapping for category filtering (from classification)
                    category_filter = self.domain_mapper.get_categories_for_domain(domain)
                    
                    if category_filter:
                        if verbose:
                            logger.info(f"[{query_id}] Using category filter: {category_filter}")
                    else:
                        if verbose:
                            logger.info(f"[{query_id}] No category filter (searching all categories)")
                    
                    # Use domain-specific top_k
                    effective_top_k = retrieval_config.top_k
                    
                    results = await self.retriever.retrieve(
                        query=query,
                        top_k=effective_top_k,
                        categories_filter=category_filter,
                        verbose=verbose,
                    )
                    
                    if results:
                        context = format_retrieval_context(results)
                        if verbose:
                            logger.info(f"[{query_id}] Retrieved {len(results)} chunks for query (domain={domain})")
                except Exception as e:
                    logger.warning(f"Retrieval failed: {e}")
            
            # Build prompt based on whether we have context
            if context:
                system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.RAG_WITH_CONTEXT)
                user_prompt = user_template.format(
                    context=context,
                    query=query,
                    temporal_hint=temporal_hint,
                    entities_hint=entities_hint,
                    choices=choices_str,
                )
            else:
                system_prompt, user_template = self.prompt_manager.get_prompt(PromptType.RAG)
                user_prompt = user_template.format(
                    query=query,
                    temporal_hint=temporal_hint,
                    entities_hint=entities_hint,
                    choices=choices_str,
                )
            
            if verbose:
                logger.info(f"[{query_id}] RAG Task User Prompt: {user_prompt}")
            response_text = await self.llm_service.generate(
                user_input=user_prompt,
                system_message=system_prompt,
            )
            if verbose:
                logger.debug(f"[{query_id}] RAG Task LLM response: {response_text}")
            
            return self._parse_json_answer(response_text, options, verbose)
            
        except Exception as e:
            logger.error(f"Error invoking RAG Task: {e}")
            return {"answer": "A"}
