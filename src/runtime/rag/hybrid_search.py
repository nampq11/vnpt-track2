"""Hybrid search engine combining BM25 and FAISS vector search (FR-07)"""

from typing import List, Optional, Tuple
import numpy as np
from src.core.models import Chunk, HybridSearchResults, SearchResult
from src.core.constants import MAX_CHUNKS_RETRIEVED, MIN_RELEVANCE_SCORE
from src.runtime.rag.fusion import RRFFusion
from src.runtime.rag.temporal_filter import TemporalFilter
from src.runtime.llm.base import LLMService


class HybridSearchEngine:
    """Combines BM25 keyword search with FAISS vector similarity search"""
    
    def __init__(
        self,
        llm_service: LLMService,
        faiss_index: Optional[object] = None,
        bm25_index: Optional[object] = None,
        chunks: Optional[List[Chunk]] = None,
    ):
        """
        Initialize hybrid search engine
        
        Args:
            llm_service: LLM service for query embedding
            faiss_index: Loaded FAISS index object
            bm25_index: Loaded BM25 index object
            chunks: List of chunks (for reference during search)
        """
        self.llm_service = llm_service
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks or []
        
        self.rrf_fusion = RRFFusion()
        self.temporal_filter = TemporalFilter()
    
    async def search(
        self,
        query: str,
        top_k: int = MAX_CHUNKS_RETRIEVED,
        temporal_year: Optional[int] = None,
    ) -> HybridSearchResults:
        """
        Perform hybrid search combining BM25 and vector similarity
        
        Process:
        1. BM25 keyword search
        2. Vector semantic search (via FAISS)
        3. Temporal filtering (if year provided)
        4. RRF fusion of both results
        
        Args:
            query: Search query
            top_k: Number of top results to return
            temporal_year: Optional year for temporal filtering
            
        Returns:
            HybridSearchResults with ranked chunks
        """
        # Step 1: BM25 search
        bm25_results = await self._bm25_search(query, top_k)
        
        # Step 2: Vector search
        vector_results = await self._vector_search(query, top_k)
        
        # Step 3: Temporal filtering
        temporal_filter_applied = False
        if temporal_year is not None:
            bm25_results = [
                (chunk, score) for chunk, score in bm25_results
                if chunk.metadata.valid_from <= temporal_year <= chunk.metadata.expire_at
            ]
            vector_results = [
                (chunk, score) for chunk, score in vector_results
                if chunk.metadata.valid_from <= temporal_year <= chunk.metadata.expire_at
            ]
            temporal_filter_applied = True
        
        # Step 4: RRF fusion
        fused_results = self.rrf_fusion.fuse(bm25_results, vector_results, top_k)
        
        # Filter by minimum relevance score
        fused_results = [
            r for r in fused_results
            if r.final_score >= MIN_RELEVANCE_SCORE
        ]
        
        return HybridSearchResults(
            results=fused_results,
            total_retrieved=len(fused_results),
            temporal_filter_applied=temporal_filter_applied,
        )
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """
        BM25 keyword search using bm25s library
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25_index is None or not self.chunks:
            return []
        
        try:
            # Tokenize query (same method as used in BM25IndexBuilder)
            import re
            query_lower = query.lower()
            query_tokens = re.findall(r'\w+', query_lower, flags=re.UNICODE)
            
            if not query_tokens:
                return []
            
            # BM25 search using bm25s library
            # The index returns (indices, scores) for top-k results
            indices, scores = self.bm25_index.retrieve(query_tokens, k=top_k)
            
            # Map indices to chunks with scores
            results = []
            for idx, score in zip(indices, scores):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            
            return results
        except Exception as e:
            print(f"BM25 search error: {str(e)}")
            return []
    
    async def _vector_search(
        self,
        query: str,
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        """
        Vector similarity search using FAISS
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        if self.faiss_index is None or not self.chunks:
            return []
        
        try:
            # Get query embedding
            query_embedding = await self.llm_service.embed(query)
            
            if not query_embedding:
                return []
            
            query_vector = np.array([query_embedding]).astype(np.float32)
            
            # Search FAISS index
            # FAISS returns distances, we convert to similarity scores
            distances, indices = self.faiss_index.search(query_vector, top_k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.chunks):
                    # Convert distance to similarity (lower distance = higher similarity)
                    similarity = 1.0 / (1.0 + float(distance))
                    results.append((self.chunks[idx], similarity))
            
            return results
        except Exception as e:
            print(f"Vector search error: {str(e)}")
            return []
    
    def load_faiss_index(self, index_path: str) -> None:
        """
        Load FAISS index from file
        
        Args:
            index_path: Path to faiss.index file
        """
        try:
            import faiss
            self.faiss_index = faiss.read_index(index_path)
        except Exception as e:
            print(f"Failed to load FAISS index: {str(e)}")
    
    def load_bm25_index(self, index_path: str) -> None:
        """
        Load BM25 index from file
        
        Args:
            index_path: Path to bm25.pkl file
        """
        try:
            import pickle
            with open(index_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
        except Exception as e:
            print(f"Failed to load BM25 index: {str(e)}")
    
    def load_chunks(self, chunks: List[Chunk]) -> None:
        """
        Load chunks for reference
        
        Args:
            chunks: List of chunks
        """
        self.chunks = chunks

