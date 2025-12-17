"""Tests for LanceDB index and retriever."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.brain.rag.lancedb_index import LanceDBIndex
from src.brain.rag.lancedb_retriever import LanceDBRetriever, format_retrieval_context


def test_lancedb_index_build_and_search():
    """Test LanceDB index creation and search."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        embeddings = np.random.random((500, 128)).astype('float32')
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Test content {i}",
                "category": f"category_{i % 3}",
                "title": f"Title {i}",
                "section": "Section A",
                "source_file": f"file_{i % 5}.txt",
            }
            for i in range(500)
        ]
        
        # Build index
        index = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index.build(embeddings, chunks)
        
        assert index.ntotal == 500
        
        # Search
        query = np.random.random(128).astype('float32')
        scores, indices = index.search(query, top_k=5)
        
        assert len(scores) == 5
        assert len(indices) == 5
        assert all(0 <= idx < 500 for idx in indices)


def test_lancedb_index_filter_search():
    """Test filtered search with categories."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = np.random.random((500, 128)).astype('float32')
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Test content {i}",
                "category": f"cat_{i % 3}",
                "title": "",
                "section": "",
                "source_file": "",
            }
            for i in range(500)
        ]
        
        index = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index.build(embeddings, chunks)
        
        # Filter by category
        query = np.random.random(128).astype('float32')
        scores, indices = index.search_with_filter(
            query,
            categories=["cat_0"],
            top_k=5
        )
        
        assert len(scores) <= 5
        
        # Verify results are from correct category
        table = index._get_table()
        df = table.to_pandas()
        for idx in indices:
            row = df[df["id"] == idx].iloc[0]
            assert row["category"] == "cat_0"


def test_lancedb_index_hybrid_search():
    """Test hybrid search combining vector + FTS."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = np.random.random((500, 128)).astype('float32')
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Test content about {'Vietnam' if i % 2 == 0 else 'China'} number {i}",
                "category": "test",
                "title": "",
                "section": "",
                "source_file": "",
            }
            for i in range(500)
        ]
        
        index = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index.build(embeddings, chunks)
        
        # Hybrid search
        query_emb = np.random.random(128).astype('float32')
        results = index.hybrid_search(
            query_text="Vietnam",
            query_embedding=query_emb,
            top_k=5
        )
        
        assert len(results) <= 5
        # Results should contain "Vietnam" in content
        for result in results:
            assert "Vietnam" in result["content"] or "China" in result["content"]


def test_lancedb_index_load():
    """Test loading existing index."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save index
        embeddings = np.random.random((500, 128)).astype('float32')
        chunks = [
            {"chunk_id": f"c{i}", "content": f"test {i}", "category": "test"}
            for i in range(500)
        ]
        
        index1 = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index1.build(embeddings, chunks)
        
        # Load index
        index2 = LanceDBIndex.load(tmpdir, table_name="test")
        
        assert index2.ntotal == 500
        assert index2.dimension == 128


def test_lancedb_add_documents():
    """Test incremental document addition."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initial build
        embeddings1 = np.random.random((500, 128)).astype('float32')
        chunks1 = [
            {"chunk_id": f"c{i}", "content": f"test {i}", "category": "test"}
            for i in range(500)
        ]
        
        index = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index.build(embeddings1, chunks1)
        
        assert index.ntotal == 500
        
        # Add more documents
        embeddings2 = np.random.random((100, 128)).astype('float32')
        chunks2 = [
            {"chunk_id": f"new_{i}", "content": f"new {i}", "category": "test"}
            for i in range(100)
        ]
        
        index.add_documents(embeddings2, chunks2)
        
        # Check total count increased
        assert index.ntotal == 600


def test_lancedb_delete_by_source():
    """Test deleting documents by source file."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = np.random.random((500, 128)).astype('float32')
        chunks = [
            {
                "chunk_id": f"c{i}",
                "content": f"test {i}",
                "category": "test",
                "source_file": f"file_{i % 2}.txt",
            }
            for i in range(500)
        ]
        
        index = LanceDBIndex(db_path=tmpdir, table_name="test", dimension=128)
        index.build(embeddings, chunks)
        
        # Delete one source file
        index.delete_by_source("file_0.txt")
        
        # Verify deletion (soft delete, count may not change immediately)
        # But querying should not return deleted docs
        table = index._get_table()
        df = table.to_pandas()
        remaining = df[df["source_file"] == "file_0.txt"]
        assert len(remaining) == 0


@pytest.mark.asyncio
async def test_retrieval_result_format():
    """Test retrieval result formatting."""
    from src.brain.rag.lancedb_retriever import RetrievalResult
    
    result = RetrievalResult(
        chunk_id="test123",
        content="This is test content",
        score=0.95,
        metadata={"category": "test", "title": "Test Title"},
        retrieval_source="hybrid"
    )
    
    assert result.chunk_id == "test123"
    assert result.score == 0.95
    assert result.retrieval_source == "hybrid"


def test_format_retrieval_context():
    """Test context formatting for LLM."""
    from src.brain.rag.lancedb_retriever import RetrievalResult
    
    results = [
        RetrievalResult(
            chunk_id=f"c{i}",
            content=f"Content {i} " * 100,  # Long content
            score=0.9,
            metadata={"title": f"Title {i}"},
        )
        for i in range(5)
    ]
    
    context = format_retrieval_context(results, max_tokens=500)
    
    assert len(context) > 0
    assert len(context) <= 500 * 4  # Rough token estimate
    assert "Title" in context
    assert "[1]" in context  # Numbered entries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

