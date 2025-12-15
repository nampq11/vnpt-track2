"""Reciprocal Rank Fusion (RRF) for combining search results"""

from typing import List, Dict, Tuple
from src.core.models import SearchResult, Chunk
from src.core.constants import BM25_WEIGHT, VECTOR_WEIGHT


class RRFFusion:
    """Combines BM25 and vector search results using Reciprocal Rank Fusion"""
    
    def __init__(
        self,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
        k: int = 60,  # RRF constant
    ):
        """
        Initialize RRF fusion
        
        Args:
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector search scores
            k: RRF constant (typical value 60)
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.k = k
    
    def fuse(
        self,
        bm25_results: List[Tuple[Chunk, float]],
        vector_results: List[Tuple[Chunk, float]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Fuse BM25 and vector search results
        
        Uses Reciprocal Rank Fusion (RRF) formula:
        RRF(d) = Σ 1/(k + rank)
        
        Args:
            bm25_results: List of (chunk, score) tuples from BM25
            vector_results: List of (chunk, score) tuples from vector search
            top_k: Number of top results to return
            
        Returns:
            Sorted list of SearchResult objects
        """
        # Build ranking dictionaries by chunk ID
        bm25_rankings: Dict[str, Tuple[Chunk, float, int]] = {}
        vector_rankings: Dict[str, Tuple[Chunk, float, int]] = {}
        
        # Assign ranks (1-indexed)
        for rank, (chunk, score) in enumerate(bm25_results, 1):
            bm25_rankings[chunk.id] = (chunk, score, rank)
        
        for rank, (chunk, score) in enumerate(vector_results, 1):
            vector_rankings[chunk.id] = (chunk, score, rank)
        
        # Collect all chunk IDs
        all_ids = set(bm25_rankings.keys()) | set(vector_rankings.keys())
        
        # Calculate fused scores
        fused: Dict[str, SearchResult] = {}
        
        for chunk_id in all_ids:
            bm25_tuple = bm25_rankings.get(chunk_id)
            vector_tuple = vector_rankings.get(chunk_id)
            
            # Get chunk and scores
            chunk = bm25_tuple[0] if bm25_tuple else vector_tuple[0]
            bm25_score = bm25_tuple[1] if bm25_tuple else 0.0
            vector_score = vector_tuple[1] if vector_tuple else 0.0
            bm25_rank = bm25_tuple[2] if bm25_tuple else len(bm25_results) + 1
            vector_rank = vector_tuple[2] if vector_tuple else len(vector_results) + 1
            
            # Calculate RRF scores
            bm25_rrf = 1.0 / (self.k + bm25_rank)
            vector_rrf = 1.0 / (self.k + vector_rank)
            
            # Weighted combination
            final_score = (
                self.bm25_weight * bm25_rrf +
                self.vector_weight * vector_rrf
            )
            
            fused[chunk_id] = SearchResult(
                chunk=chunk,
                keyword_score=bm25_score,
                semantic_score=vector_score,
                final_score=final_score,
                rank=0,  # Will be set after sorting
            )
        
        # Sort by final score descending
        sorted_results = sorted(
            fused.values(),
            key=lambda r: r.final_score,
            reverse=True,
        )
        
        # Assign final ranks and truncate
        for rank, result in enumerate(sorted_results[:top_k], 1):
            result.rank = rank
        
        return sorted_results[:top_k]
    
    @staticmethod
    def normalize_scores(scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range
        
        Args:
            scores: List of scores
            
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]

