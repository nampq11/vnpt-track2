"""BM25 lexical search index for Vietnamese documents."""

import pickle
import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi
from loguru import logger

try:
    from underthesea import word_tokenize
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    logger.warning("underthesea not available, falling back to simple tokenization")
    UNDERTHESEA_AVAILABLE = False

from src.brain.rag.document_processor import DocumentChunk


class BM25Index:
    """BM25 lexical search index for document chunks."""
    
    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        
    def build(self, chunks: List[DocumentChunk]):
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.content) 
            for chunk in chunks
        ]
        
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=1.5,      # Term frequency saturation
            b=0.75,      # Document length normalization
            epsilon=0.25 # IDF floor
        )
        
        logger.info(f"Built BM25 index with {len(chunks)} documents")
        logger.info(f"Average document length: {self.bm25.avgdl:.1f} tokens")
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Vietnamese tokenization using underthesea.
        
        Falls back to simple word split if underthesea not available.
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(
            r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]',
            ' ',
            text
        )
        
        if UNDERTHESEA_AVAILABLE:
            # Use underthesea for Vietnamese word segmentation
            try:
                tokens = word_tokenize(text, format="text").split()
            except Exception as e:
                logger.warning(f"Underthesea tokenization failed: {e}, using fallback")
                tokens = text.split()
        else:
            # Fallback: simple word split
            tokens = text.split()
        
        # Remove very short tokens (likely noise)
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            doc_indices: Optional list of indices to search within (for pre-filtering)
            
        Returns:
            List of (chunk_index, score) tuples
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        
        tokenized_query = self._tokenize(query)
        
        if doc_indices is not None:
            # Search within subset of documents
            scores = self.bm25.get_batch_scores(tokenized_query, doc_indices)
            results = list(zip(doc_indices, scores))
        else:
            # Search all documents
            scores = self.bm25.get_scores(tokenized_query)
            results = list(enumerate(scores))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def get_top_chunks(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[DocumentChunk, float]]:
        """Get top matching chunks with their scores."""
        results = self.search(query, top_k)
        return [(self.chunks[idx], score) for idx, score in results]
    
    def save(self, path: str):
        """Save BM25 index to file."""
        data = {
            "bm25": self.bm25,
            "tokenized_corpus": self.tokenized_corpus,
            # Note: chunks are saved separately in chunks.json
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {path}")
    
    @classmethod
    def load(cls, path: str, chunks: List[DocumentChunk]) -> "BM25Index":
        """Load BM25 index from file."""
        instance = cls()
        instance.chunks = chunks
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        instance.bm25 = data["bm25"]
        instance.tokenized_corpus = data["tokenized_corpus"]
        
        logger.info(f"Loaded BM25 index from {path}")
        return instance


def build_bm25_index(chunks: List[DocumentChunk], output_path: str) -> BM25Index:
    """Build and save BM25 index."""
    index = BM25Index()
    index.build(chunks)
    index.save(output_path)
    return index

