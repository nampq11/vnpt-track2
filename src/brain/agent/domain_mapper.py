"""Domain-to-Category Mapping for RAG Retrieval.

Maps high-level domain types to specific data categories for intelligent filtering.
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
from src.models.tasks.rag import DomainRAGTask
from loguru import logger


@dataclass
class RetrievalConfig:
    """Domain-specific retrieval configuration."""
    top_k: int = 5
    vector_weight: float = 0.6  # Weight for vector similarity
    fts_weight: float = 0.4  # Weight for full-text search
    use_temporal_filter: bool = False
    temporal_boost: float = 1.2  # Boost score for temporal matches


# Comprehensive mapping from domain to data categories
DOMAIN_CATEGORY_MAPPING: Dict[DomainRAGTask, List[str]] = {
    DomainRAGTask.LAW: [
        "Phap_luat_Viet_Nam",
        "hien_phap",
        "Quyen_nghia_vu",
        "cong_hoa_xa_hoi_chu_nghia_VN",  # Government structure/legal framework
        "dang_cong_san_viet_nam",  # Party regulations
    ],
    
    DomainRAGTask.HISTORY: [
        "Lich_Su_Viet_nam",
        "Bac_Ho",
        "khang_chien_lon",
        "nhan_vat_lich_su_tieu_bieu",
        "nhan_vat_chinh_tri",
        "Tu_Tuong_HCM",
        "dang_cong_san_viet_nam",  # Historical context
    ],
    
    DomainRAGTask.GEOGRAPHY: [
        "Dia_ly_viet_nam",
        "Dia_chinh_Viet_nam",
        "Dia_hinh_Viet_Nam",
        "Dia_dien_du_lich",
        "Khi_hau_thoi_tiet_thien_tai",
        "song_ngoi_bien_tai_nguyen_thien_nhien",
    ],
    
    DomainRAGTask.CULTURE: [
        "Van_Hoa_Viet_Nam",
        "Van_hoa_am_thuc",
        "van_hoa_lang_xa",
        "Van_hoa_ung_xu",
        "Phong_tuc_tap_quan",
        "le_hoi_truyen_thong",
        "tin_nguong_ton_giao",
    ],
    
    DomainRAGTask.POLITICS: [
        "cong_hoa_xa_hoi_chu_nghia_VN",
        "dang_cong_san_viet_nam",
        "nhan_vat_chinh_tri",
        "Quoc_phong_Viet_nam",
        "Bac_Ho",  # Political figure
        "Tu_Tuong_HCM",  # Political ideology
    ],
    
    # GENERAL_KNOWLEDGE: No filtering, search across all categories
    DomainRAGTask.GENERAL_KNOWLEDGE: None,
}


# Domain-specific retrieval configurations
DOMAIN_RETRIEVAL_CONFIG: Dict[DomainRAGTask, RetrievalConfig] = {
    DomainRAGTask.LAW: RetrievalConfig(
        top_k=5,
        vector_weight=0.7,
        fts_weight=0.3,
        use_temporal_filter=True,
        temporal_boost=1.3,
    ),
    
    DomainRAGTask.HISTORY: RetrievalConfig(
        top_k=7,
        vector_weight=0.6,
        fts_weight=0.4,
        use_temporal_filter=True,
        temporal_boost=1.5,
    ),
    
    DomainRAGTask.GEOGRAPHY: RetrievalConfig(
        top_k=3,
        vector_weight=0.8,
        fts_weight=0.2,
        use_temporal_filter=False,
        temporal_boost=1.0,
    ),
    
    DomainRAGTask.CULTURE: RetrievalConfig(
        top_k=5,
        vector_weight=0.5,
        fts_weight=0.5,
        use_temporal_filter=False,
        temporal_boost=1.1,
    ),
    
    DomainRAGTask.POLITICS: RetrievalConfig(
        top_k=5,
        vector_weight=0.65,
        fts_weight=0.35,
        use_temporal_filter=True,
        temporal_boost=1.2,
    ),
    
    DomainRAGTask.GENERAL_KNOWLEDGE: RetrievalConfig(
        top_k=5,
        vector_weight=0.6,
        fts_weight=0.4,
        use_temporal_filter=False,
        temporal_boost=1.0,
    ),
}


class DomainMapper:
    """Maps domains to categories and retrieval configurations."""
    
    @staticmethod
    def get_categories_for_domain(domain: DomainRAGTask) -> Optional[List[str]]:
        """
        Get list of data categories for a given domain.
        
        Args:
            domain: Domain type from classification
            
        Returns:
            List of category names, or None for no filtering (GENERAL_KNOWLEDGE)
        """
        categories = DOMAIN_CATEGORY_MAPPING.get(domain)
        
        if categories:
            logger.debug(f"Domain {domain} mapped to {len(categories)} categories: {categories[:3]}...")
        else:
            logger.debug(f"Domain {domain} → no category filter (search all)")
        
        return categories
    
    @staticmethod
    def get_retrieval_config(domain: DomainRAGTask) -> RetrievalConfig:
        """
        Get domain-specific retrieval configuration.
        
        Args:
            domain: Domain type from classification
            
        Returns:
            RetrievalConfig with optimized parameters for the domain
        """
        config = DOMAIN_RETRIEVAL_CONFIG.get(
            domain,
            DOMAIN_RETRIEVAL_CONFIG[DomainRAGTask.GENERAL_KNOWLEDGE]
        )
        
        logger.debug(
            f"Retrieval config for {domain}: "
            f"top_k={config.top_k}, "
            f"weights={config.vector_weight:.1f}/{config.fts_weight:.1f}, "
            f"temporal={'ON' if config.use_temporal_filter else 'OFF'}"
        )
        
        return config
    
    @staticmethod
    def merge_with_entity_categories(
        domain_categories: Optional[List[str]],
        entity_categories: Optional[List[str]],
    ) -> Optional[List[str]]:
        """
        Merge domain-based categories with entity-based categories.
        
        Strategy:
        - If domain_categories is None (GENERAL_KNOWLEDGE), use entity_categories
        - If entity_categories is None, use domain_categories
        - If both exist, use intersection (more precise) or union (more recall)
        
        Args:
            domain_categories: Categories from domain mapping
            entity_categories: Categories from entity extraction
            
        Returns:
            Merged category list, or None for no filtering
        """
        # Priority 1: Domain-based categories
        if domain_categories is not None and len(domain_categories) > 0:
            # If we also have entity categories, take intersection for precision
            if entity_categories and len(entity_categories) > 0:
                intersection = list(set(domain_categories) & set(entity_categories))
                if intersection:
                    logger.debug(
                        f"Using intersection of domain + entity categories: {intersection}"
                    )
                    return intersection
                else:
                    # No overlap, trust domain more than entities
                    logger.debug(
                        f"No category overlap, using domain categories: {domain_categories[:3]}..."
                    )
                    return domain_categories
            else:
                return domain_categories
        
        # Priority 2: Entity-based categories (fallback)
        if entity_categories and len(entity_categories) > 0:
            logger.debug(f"Using entity-based categories: {entity_categories}")
            return entity_categories
        
        # Priority 3: No filtering
        logger.debug("No category filter applied (searching all)")
        return None

