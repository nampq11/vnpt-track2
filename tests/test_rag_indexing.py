"""Test RAG indexing system components."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.brain.rag.document_processor import DocumentProcessor, DocumentChunk


def test_document_processor():
    """Test document processing and chunking."""
    processor = DocumentProcessor(chunk_size=512, overlap=50)
    
    # Test chunking
    text = "This is a test document. " * 50  # Create a long text
    metadata = {"category": "test", "title": "Test", "section": "Test Section"}
    
    chunks = processor._chunk_text(text, metadata)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    # Chunks can be slightly longer due to sentence boundary alignment
    assert all(len(chunk.content) <= 650 for chunk in chunks)
    assert all(chunk.metadata == metadata for chunk in chunks)
    
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Chunk lengths: min={min(len(c.content) for c in chunks)}, max={max(len(c.content) for c in chunks)}")


def test_document_processor_with_real_data():
    """Test processing real data directory."""
    processor = DocumentProcessor(chunk_size=512, overlap=50)
    data_dir = Path("data/data")
    
    if not data_dir.exists():
        pytest.skip("Data directory not found")
    
    # Process a single category
    test_dir = data_dir / "Bac_Ho"
    if not test_dir.exists():
        pytest.skip("Test directory not found")
    
    chunks = []
    for file_path in test_dir.glob("*.txt"):
        file_chunks = processor._process_file(file_path, "Bac_Ho")
        chunks.extend(file_chunks)
        break  # Process just one file
    
    assert len(chunks) > 0
    print(f"✓ Processed file into {len(chunks)} chunks")
    print(f"  Sample chunk: {chunks[0].content[:100]}...")


def test_underthesea_tokenization():
    """Test underthesea tokenization."""
    try:
        from src.brain.rag.bm25_index import BM25Index
        
        index = BM25Index()
        text = "Hồ Chí Minh là vị lãnh tụ vĩ đại của dân tộc Việt Nam"
        tokens = index._tokenize(text)
        
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        
        print(f"✓ Tokenized: {text}")
        print(f"  Tokens: {tokens}")
        
    except ImportError:
        pytest.skip("underthesea not available")


if __name__ == "__main__":
    print("Running RAG indexing tests...\n")
    
    test_document_processor()
    test_document_processor_with_real_data()
    test_underthesea_tokenization()
    
    print("\n✅ All tests passed!")

