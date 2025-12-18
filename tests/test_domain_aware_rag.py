"""Test domain-aware RAG system with domain mapper."""

import pytest

from src.models.tasks.rag import DomainRAGTask
from src.brain.agent.domain_mapper import DomainMapper, DOMAIN_CATEGORY_MAPPING


class TestDomainMapper:
    """Test domain-to-category mapping functionality."""
    
    def test_get_categories_for_law_domain(self):
        """Test that LAW domain returns legal categories."""
        categories = DomainMapper.get_categories_for_domain(DomainRAGTask.LAW)
        assert categories is not None
        assert isinstance(categories, list)
        assert "Phap_luat_Viet_Nam" in categories
        assert "hien_phap" in categories
        assert len(categories) > 0
    
    def test_get_categories_for_history_domain(self):
        """Test that HISTORY domain returns historical categories."""
        categories = DomainMapper.get_categories_for_domain(DomainRAGTask.HISTORY)
        assert categories is not None
        assert "Lich_Su_Viet_nam" in categories
        assert "Bac_Ho" in categories
        assert "khang_chien_lon" in categories
    
    def test_get_categories_for_geography_domain(self):
        """Test that GEOGRAPHY domain returns geographic categories."""
        categories = DomainMapper.get_categories_for_domain(DomainRAGTask.GEOGRAPHY)
        assert categories is not None
        assert "Dia_ly_viet_nam" in categories
        assert "Dia_chinh_Viet_nam" in categories
    
    def test_get_categories_for_culture_domain(self):
        """Test that CULTURE domain returns cultural categories."""
        categories = DomainMapper.get_categories_for_domain(DomainRAGTask.CULTURE)
        assert categories is not None
        assert "Van_Hoa_Viet_Nam" in categories
        assert "Van_hoa_am_thuc" in categories
        assert "le_hoi_truyen_thong" in categories
    
    def test_get_categories_for_politics_domain(self):
        """Test that POLITICS domain returns political categories."""
        categories = DomainMapper.get_categories_for_domain(DomainRAGTask.POLITICS)
        assert categories is not None
        assert "cong_hoa_xa_hoi_chu_nghia_VN" in categories
        assert "dang_cong_san_viet_nam" in categories
        assert "nhan_vat_chinh_tri" in categories
        assert "Quoc_phong_Viet_nam" in categories
    
    def test_get_categories_for_general_knowledge_domain(self):
        """Test that GENERAL_KNOWLEDGE returns None (no filtering)."""
        categories = DomainMapper.get_categories_for_domain(
            DomainRAGTask.GENERAL_KNOWLEDGE
        )
        assert categories is None
    
    def test_get_retrieval_config_for_law(self):
        """Test LAW domain retrieval config."""
        config = DomainMapper.get_retrieval_config(DomainRAGTask.LAW)
        assert config.top_k == 5
        assert config.vector_weight == 0.7
        assert config.fts_weight == 0.3
        assert config.use_temporal_filter is True
        assert config.temporal_boost > 1.0
    
    def test_get_retrieval_config_for_history(self):
        """Test HISTORY domain retrieval config."""
        config = DomainMapper.get_retrieval_config(DomainRAGTask.HISTORY)
        assert config.top_k == 7
        assert config.use_temporal_filter is True
        assert config.temporal_boost > 1.0
    
    def test_get_retrieval_config_for_geography(self):
        """Test GEOGRAPHY domain retrieval config."""
        config = DomainMapper.get_retrieval_config(DomainRAGTask.GEOGRAPHY)
        assert config.top_k == 3
        assert config.vector_weight == 0.8  # High vector weight for geographic queries
        assert config.use_temporal_filter is False
    
    def test_get_retrieval_config_for_politics(self):
        """Test POLITICS domain retrieval config."""
        config = DomainMapper.get_retrieval_config(DomainRAGTask.POLITICS)
        assert config.top_k == 5
        assert config.vector_weight == 0.65
        assert config.fts_weight == 0.35
        assert config.use_temporal_filter is True
        assert config.temporal_boost > 1.0
    
    def test_merge_categories_domain_priority(self):
        """Test that domain categories take priority over entity categories."""
        domain_cats = ["Phap_luat_Viet_Nam", "hien_phap"]
        entity_cats = ["Lich_Su_Viet_nam", "Bac_Ho"]
        
        merged = DomainMapper.merge_with_entity_categories(
            domain_categories=domain_cats,
            entity_categories=entity_cats,
        )
        
        # Should use domain categories (no overlap)
        assert merged == domain_cats
    
    def test_merge_categories_intersection(self):
        """Test intersection when both domain and entity categories exist."""
        domain_cats = ["Phap_luat_Viet_Nam", "hien_phap", "Lich_Su_Viet_nam"]
        entity_cats = ["Lich_Su_Viet_nam", "Bac_Ho"]
        
        merged = DomainMapper.merge_with_entity_categories(
            domain_categories=domain_cats,
            entity_categories=entity_cats,
        )
        
        # Should take intersection for precision
        assert merged == ["Lich_Su_Viet_nam"]
    
    def test_merge_categories_entity_fallback(self):
        """Test fallback to entity categories when domain is None."""
        entity_cats = ["Lich_Su_Viet_nam", "Bac_Ho"]
        
        merged = DomainMapper.merge_with_entity_categories(
            domain_categories=None,
            entity_categories=entity_cats,
        )
        
        assert merged == entity_cats
    
    def test_merge_categories_no_filter(self):
        """Test no filtering when both are None."""
        merged = DomainMapper.merge_with_entity_categories(
            domain_categories=None,
            entity_categories=None,
        )
        
        assert merged is None
    
    def test_all_domains_have_mappings(self):
        """Test that all domain types have mappings defined."""
        for domain in DomainRAGTask:
            categories = DomainMapper.get_categories_for_domain(domain)
            # GENERAL_KNOWLEDGE can be None, others should have categories
            if domain != DomainRAGTask.GENERAL_KNOWLEDGE:
                assert categories is not None
                assert len(categories) > 0
    
    def test_all_domains_have_retrieval_configs(self):
        """Test that all domain types have retrieval configs."""
        for domain in DomainRAGTask:
            config = DomainMapper.get_retrieval_config(domain)
            assert config is not None
            assert config.top_k > 0
            assert 0 <= config.vector_weight <= 1
            assert 0 <= config.fts_weight <= 1
            # Weights should sum to 1.0
            assert abs(config.vector_weight + config.fts_weight - 1.0) < 0.01


@pytest.mark.asyncio
async def test_rag_task_uses_domain_mapper():
    """Integration test: RAGTask should use domain mapper."""
    from src.brain.agent.tasks.rag import RAGTask
    from src.brain.llm.services.vnpt import VNPTService
    
    # Mock LLM service
    llm_service = VNPTService(model="vnptai-hackathon-large")
    
    # Create RAG task
    rag_task = RAGTask(
        llm_service=llm_service,
        use_retrieval=False,  # Disable retrieval for unit test
    )
    
    # Verify domain mapper is initialized
    assert hasattr(rag_task, 'domain_mapper')
    assert isinstance(rag_task.domain_mapper, DomainMapper)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

