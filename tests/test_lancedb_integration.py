"""Integration tests for LanceDB with RAG task."""

import pytest
import asyncio
from src.brain.agent.tasks.rag import RAGTask
from src.brain.llm.services.vnpt import VNPTService


@pytest.mark.asyncio
async def test_rag_task_with_lancedb():
    """Test RAG task can load and use LanceDB retriever."""
    
    # Initialize service
    llm_service = VNPTService(model_type="small")
    
    # Initialize RAG task (should load LanceDB retriever)
    rag_task = RAGTask(
        llm_service=llm_service,
        use_retrieval=True,
        retrieval_top_k=3,
        index_dir="data/embeddings/knowledge",
    )
    
    # Check retriever loaded
    assert rag_task.retriever is not None, "Retriever should be loaded"
    
    # Check it's LanceDB retriever
    from src.brain.rag.lancedb_retriever import LanceDBRetriever
    assert isinstance(rag_task.retriever, LanceDBRetriever), "Should use LanceDBRetriever"
    
    print("✅ RAG task successfully loaded LanceDB retriever")


@pytest.mark.asyncio
async def test_lancedb_retrieval():
    """Test retrieval with LanceDB."""
    from src.brain.rag.lancedb_retriever import LanceDBRetriever
    from src.brain.llm.services.vnpt import VNPTService
    
    llm_service = VNPTService(model_type="embedding")
    
    # Load retriever
    retriever = LanceDBRetriever.from_directory(
        index_dir="data/embeddings/knowledge",
        llm_service=llm_service,
    )
    
    # Test retrieval
    results = await retriever.retrieve(
        query="Hồ Chí Minh sinh năm nào?",
        top_k=3,
    )
    
    assert len(results) > 0, "Should retrieve some results"
    assert results[0].content, "Results should have content"
    assert results[0].chunk_id, "Results should have chunk_id"
    
    print(f"✅ Retrieved {len(results)} results")
    print(f"   Top result: {results[0].chunk_id}")
    print(f"   Score: {results[0].score:.4f}")
    print(f"   Content preview: {results[0].content[:80]}...")


@pytest.mark.asyncio
async def test_lancedb_category_filtering():
    """Test category filtering with LanceDB."""
    from src.brain.rag.lancedb_retriever import LanceDBRetriever
    from src.brain.llm.services.vnpt import VNPTService
    
    llm_service = VNPTService(model_type="embedding")
    
    retriever = LanceDBRetriever.from_directory(
        index_dir="data/embeddings/knowledge",
        llm_service=llm_service,
    )
    
    # Test with category filter
    results = await retriever.retrieve(
        query="Hồ Chí Minh",
        top_k=5,
        categories_filter=["Bac_Ho", "Lich_Su_Viet_nam"],
    )
    
    assert len(results) > 0, "Should retrieve filtered results"
    
    # Check categories
    for result in results:
        category = result.metadata.get("category", "")
        assert category in ["Bac_Ho", "Lich_Su_Viet_nam", ""], \
            f"Result should be from filtered categories, got: {category}"
    
    print(f"✅ Category filtering works: {len(results)} results")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_rag_task_with_lancedb())
    asyncio.run(test_lancedb_retrieval())
    asyncio.run(test_lancedb_category_filtering())

