"""BM25 sparse index builder"""

from typing import List
import pickle
from src.core.models import Chunk


class BM25IndexBuilder:
    """Builds BM25 keyword index from chunks"""
    
    def __init__(self):
        """Initialize BM25 index builder"""
        self.bm25 = None
        self.corpus = None
    
    def build(self, chunks: List[Chunk]) -> object:
        """
        Build BM25 index from chunks
        
        Args:
            chunks: List of chunks to index
            
        Returns:
            BM25 index object
        """
        try:
            import bm25s
        except ImportError:
            raise ImportError("bm25s is required. Install with: pip install bm25s")
        
        # Prepare corpus (tokenized documents)
        corpus = []
        for chunk in chunks:
            # Simple Vietnamese tokenization (split by whitespace and punctuation)
            tokens = self._tokenize_vietnamese(chunk.text)
            corpus.append(tokens)
        
        # Train BM25
        self.bm25 = bm25s.BM25(corpus=corpus)
        self.corpus = corpus
        
        return self.bm25
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Search using BM25 index
        
        Args:
            query: Search query
            top_k: Number of top results
            
        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Tokenize query
        tokens = self._tokenize_vietnamese(query)
        
        # Search
        results = self.bm25.retrieve(tokens, k=top_k)
        
        return results
    
    def save(self, path: str) -> None:
        """
        Save BM25 index to file
        
        Args:
            path: Output file path
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.bm25, f)
            print(f"BM25 index saved to {path}")
        except Exception as e:
            print(f"Failed to save BM25 index: {str(e)}")
    
    @staticmethod
    def _tokenize_vietnamese(text: str) -> List[str]:
        """
        Tokenize Vietnamese text
        
        Simple tokenization by splitting on whitespace and punctuation.
        For production, consider using underthesea or PyVi.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\w+', text, flags=re.UNICODE)
        
        return tokens

