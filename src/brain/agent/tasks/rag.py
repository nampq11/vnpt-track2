from src.models.tasks.rag import DomainRAGTask
from src.brain.llm.services.type import LLMService
from src.brain.llm.services.vnpt import VNPTService
from loguru import logger
from src.brain.agent.tasks.base import BaseTask
from typing import Dict, List, Optional
from src.brain.agent.prompts import RAG_PROMPT, RAG_PROMPT_WITH_CONTEXT
from src.brain.agent.domain_mapper import DomainMapper
import re
import json
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

    def _infer_category_filter(
        self,
        key_entities: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Infer relevant categories from extracted entities.
        
        Uses entities from classification service to map to data categories.
        """
        if not key_entities:
            return None
        
        # Map entity keywords to data categories
        category_keywords = {
            "Lich_Su_Viet_nam": ["lịch sử", "chiến tranh", "kháng chiến", "thời kỳ", "triều đại", "cách mạng", "khởi nghĩa"],
            "Bac_Ho": ["Hồ Chí Minh", "Bác Hồ", "Nguyễn Ái Quốc", "Chủ tịch Hồ Chí Minh"],
            "Dia_ly_viet_nam": ["địa lý", "vùng", "miền", "tỉnh", "thành phố", "đồng bằng", "núi", "sông"],
            "Dia_chinh_Viet_nam": ["hành chính", "quận", "huyện", "xã", "phường", "thị xã"],
            "Van_Hoa_Viet_Nam": ["văn hóa", "nghệ thuật", "truyền thống", "phong tục", "lễ hội", "di sản"],
            "Van_hoa_am_thuc": ["ẩm thực", "món ăn", "đặc sản", "nấu ăn"],
            "Phap_luat_Viet_Nam": ["luật", "pháp luật", "quy định", "nghị định", "điều", "khoản", "bộ luật"],
            "dang_cong_san_viet_nam": ["Đảng", "cộng sản", "Đại hội", "Trung ương", "BCH", "Ban Chấp hành"],
            "Kinh_Te_Viet_Nam": ["kinh tế", "GDP", "thương mại", "công nghiệp", "nông nghiệp", "xuất khẩu"],
            "khang_chien_lon": ["Điện Biên Phủ", "kháng chiến", "chiến thắng", "chiến dịch"],
            "Giao_duc": ["giáo dục", "đào tạo", "trường", "học", "sinh viên", "thi cử"],
            "Dia_dien_du_lich": ["du lịch", "danh lam", "thắng cảnh", "di tích", "vườn quốc gia", "Hạ Long", "Sapa"],
            "Quoc_phong_Viet_nam": ["quốc phòng", "an ninh", "quân đội", "lực lượng vũ trang"],
        }
        
        # Join entities to single search text
        entities_text = " ".join(key_entities).lower()
        matched_categories = []
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword.lower() in entities_text:
                    matched_categories.append(category)
                    break
        
        return matched_categories if matched_categories else None
    
    async def invoke(
        self,
        query: str,
        domain: DomainRAGTask,
        options: Dict[str, str],
        temporal_constraint: Optional[int] = None,
        key_entities: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        try:
            logger.info(f"RAG Task invoked with domain={domain}, query={query[:50]}...")
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
                    
                    # --- SMART DOMAIN-AWARE FILTERING ---
                    # Priority 1: Use domain mapping for category filtering
                    domain_categories = self.domain_mapper.get_categories_for_domain(domain)
                    
                    # Priority 2: Infer categories from entities (fallback)
                    entity_categories = self._infer_category_filter(key_entities)
                    
                    # Merge strategies for optimal filtering
                    category_filter = self.domain_mapper.merge_with_entity_categories(
                        domain_categories=domain_categories,
                        entity_categories=entity_categories,
                    )
                    
                    if category_filter:
                        logger.info(f"Using category filter: {category_filter}")
                    else:
                        logger.info("No category filter (searching all categories)")
                    
                    # Use domain-specific top_k
                    effective_top_k = retrieval_config.top_k
                    
                    results = await self.retriever.retrieve(
                        query=query,
                        top_k=effective_top_k,
                        categories_filter=category_filter,
                    )
                    
                    if results:
                        context = format_retrieval_context(results)
                        logger.info(f"Retrieved {len(results)} chunks for query (domain={domain})")
                except Exception as e:
                    logger.warning(f"Retrieval failed: {e}")
            
            # Build prompt
            if context:
                prompt = RAG_PROMPT_WITH_CONTEXT.format(
                    context=context,
                    query=query,
                    temporal_hint=temporal_hint,
                    entities_hint=entities_hint,
                    choices=choices_str,
                )
            else:
                prompt = RAG_PROMPT.format(
                    query=query,
                    temporal_hint=temporal_hint,
                    entities_hint=entities_hint,
                    choices=choices_str,
                )
            logger.info(f"RAG Task Prompt: {prompt}")
            response_text = await self.llm_service.generate(
                user_input=prompt,
            )
            logger.debug(f"RAG Task LLM response: {response_text}")
            
            return self._parse_json_answer(response_text, options)
            
        except Exception as e:
            logger.error(f"Error invoking RAG Task: {e}")
            return {"answer": "A"}