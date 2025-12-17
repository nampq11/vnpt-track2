"""Hybrid retrieval with BM25 + FAISS and RRF ranking."""

import numpy as np
import aiohttp
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
from loguru import logger
from pathlib import Path

from src.brain.llm.services.type import LLMService
from src.brain.rag.document_processor import DocumentChunk, load_chunks
from src.brain.rag.bm25_index import BM25Index
from src.brain.rag.faiss_index import FAISSIndex


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    bm25_rank: Optional[int] = None
    faiss_rank: Optional[int] = None
    retrieval_source: str = "hybrid"  # "bm25", "faiss", or "hybrid"


class HybridRetriever:
    """Hybrid BM25 + FAISS retriever with RRF ranking."""
    
    def __init__(
        self,
        faiss_index: FAISSIndex,
        bm25_index: BM25Index,
        chunks: List[DocumentChunk],
        llm_service: LLMService,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        faiss_weight: float = 1.0,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            faiss_index: FAISS vector index
            bm25_index: BM25 lexical index
            chunks: List of document chunks
            llm_service: LLM service for embedding queries
            rrf_k: RRF constant (default 60)
            bm25_weight: Weight for BM25 scores in RRF
            faiss_weight: Weight for FAISS scores in RRF
        """
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.llm_service = llm_service
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        
        # Build category index for pre-filtering
        self._category_index = self._build_category_index()
        
    def _build_category_index(self) -> Dict[str, List[int]]:
        """Build index mapping categories to chunk indices."""
        index = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            category = chunk.metadata.get("category", "unknown")
            index[category].append(i)
        return dict(index)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_top_k: int = 20,
        faiss_top_k: int = 20,
        category_filter: Optional[str] = None,
        categories_filter: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Search query
            top_k: Number of final results
            bm25_top_k: Number of BM25 candidates
            faiss_top_k: Number of FAISS candidates
            category_filter: Single category to filter (pre-filter)
            categories_filter: Multiple categories to filter (pre-filter)
            min_score: Minimum RRF score threshold (post-filter)
            
        Returns:
            List of RetrievalResult sorted by RRF score
        """
        # Determine valid indices for pre-filtering
        logger.info(
            f"Retriever retrieve with query: {query}, category_filter: {category_filter}, categories_filter: {categories_filter}"
        )
        valid_indices = None
        if category_filter:
            valid_indices = self._category_index.get(category_filter, [])
        elif categories_filter:
            valid_indices = []
            for cat in categories_filter:
                valid_indices.extend(self._category_index.get(cat, []))
            valid_indices = list(set(valid_indices))
        
        # BM25 search
        bm25_results = self.bm25_index.search(
            query=query,
            top_k=bm25_top_k,
            doc_indices=valid_indices,
        )
        
        # FAISS search
        query_embedding = await self._get_query_embedding(query)
        
        if query_embedding is not None:
            if valid_indices:
                faiss_distances, faiss_indices = self.faiss_index.search_with_filter(
                    query_embedding=query_embedding,
                    valid_indices=valid_indices,
                    top_k=faiss_top_k,
                )
            else:
                faiss_distances, faiss_indices = self.faiss_index.search(
                    query_embedding=query_embedding,
                    top_k=faiss_top_k,
                )
            faiss_results = list(zip(faiss_indices.tolist(), faiss_distances.tolist()))
        else:
            logger.warning("Failed to get query embedding, using BM25 only")
            faiss_results = []
        
        # RRF fusion
        fused_results = self._rrf_fusion(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
        )
        
        # Build result objects
        results = []
        for idx, score, bm25_rank, faiss_rank in fused_results[:top_k * 2]:  # Get more, filter later
            if score < min_score:
                continue
                
            chunk = self.chunks[idx]
            
            # Determine source
            if bm25_rank is not None and faiss_rank is not None:
                source = "hybrid"
            elif bm25_rank is not None:
                source = "bm25"
            else:
                source = "faiss"
            
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
                bm25_rank=bm25_rank,
                faiss_rank=faiss_rank,
                retrieval_source=source,
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    async def _get_query_embedding(
        self,
        query: str,
    ) -> Optional[np.ndarray]:
        """Get embedding for query."""
        try:
            async with aiohttp.ClientSession() as session:
                response = await self.llm_service.get_embedding(
                    session=session,
                    text=query,
                )
                
                # Handle different response formats
                if isinstance(response, dict) and 'data' in response:
                    # VNPT format
                    embedding = response['data'][0].get('embedding')
                elif isinstance(response, list):
                    # Azure format
                    embedding = response
                else:
                    return None
                
                return np.array(embedding, dtype='float32')
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        faiss_results: List[Tuple[int, float]],
    ) -> List[Tuple[int, float, Optional[int], Optional[int]]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: List of (doc_idx, bm25_score)
            faiss_results: List of (doc_idx, faiss_score)
            
        Returns:
            List of (doc_idx, rrf_score, bm25_rank, faiss_rank) sorted by rrf_score
        """
        scores = defaultdict(float)
        bm25_ranks = {}
        faiss_ranks = {}
        
        # Process BM25 results
        for rank, (idx, _) in enumerate(bm25_results):
            # RRF score with weight
            scores[idx] += self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
            bm25_ranks[idx] = rank + 1  # 1-indexed rank
        
        # Process FAISS results
        for rank, (idx, _) in enumerate(faiss_results):
            scores[idx] += self.faiss_weight * (1.0 / (self.rrf_k + rank + 1))
            faiss_ranks[idx] = rank + 1
        
        # Sort by RRF score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add rank info
        results_with_ranks = [
            (idx, score, bm25_ranks.get(idx), faiss_ranks.get(idx))
            for idx, score in sorted_results
        ]
        
        return results_with_ranks
    
    def get_available_categories(self) -> List[str]:
        """Return list of available categories for filtering."""
        return list(self._category_index.keys())
    
    @classmethod
    def from_directory(
        cls,
        index_dir: str,
        llm_service: LLMService,
        **kwargs,
    ) -> "HybridRetriever":
        """
        Load retriever from index directory.
        
        Args:
            index_dir: Path to directory containing index files
            llm_service: LLM service for embeddings
            **kwargs: Additional arguments for HybridRetriever
            
        Returns:
            Initialized HybridRetriever
        """
        index_path = Path(index_dir)
        
        # Load chunks
        chunks_path = index_path / "chunks.json"
        chunks = load_chunks(str(chunks_path))
        
        # Load FAISS index
        faiss_path = index_path / "faiss.index"
        faiss_index = FAISSIndex.load(str(faiss_path))
        
        # Load BM25 index
        bm25_path = index_path / "bm25.pkl"
        bm25_index = BM25Index.load(str(bm25_path), chunks)
        
        logger.info(f"Loaded retriever from {index_dir}")
        logger.info(f"  - Chunks: {len(chunks)}")
        logger.info(f"  - FAISS vectors: {faiss_index.ntotal}")
        
        return cls(
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            chunks=chunks,
            llm_service=llm_service,
            **kwargs,
        )


def format_retrieval_context(
    results: List[RetrievalResult],
    max_tokens: int = 2000,
) -> str:
    """
    Format retrieval results as context for LLM.
    
    Args:
        results: List of retrieval results
        max_tokens: Approximate max tokens (chars / 4)
        
    Returns:
        Formatted context string
    """
    context_parts = []
    total_chars = 0
    max_chars = max_tokens * 4  # Rough token estimate
    
    for i, result in enumerate(results, 1):
        part = f"[{i}] {result.metadata.get('title', 'Unknown')}\n{result.content}\n"
        
        if total_chars + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_chars += len(part)
    
    return "\n---\n".join(context_parts)

