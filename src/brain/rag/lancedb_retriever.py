"""LanceDB-native hybrid retriever with built-in RRF reranking."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from loguru import logger
import numpy as np
import aiohttp
from pathlib import Path

from src.brain.llm.services.type import LLMService
from src.brain.rag.lancedb_index import LanceDBIndex


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_source: str = "hybrid"


class LanceDBRetriever:
    """LanceDB-native hybrid retriever with built-in RRF reranking."""
    
    def __init__(
        self,
        lancedb_index: LanceDBIndex,
        llm_service: LLMService,
    ):
        """
        Initialize LanceDB retriever.
        
        Args:
            lancedb_index: LanceDB index instance
            llm_service: LLM service for embedding queries
        """
        self.index = lancedb_index
        self.llm_service = llm_service
        
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        categories_filter: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using LanceDB native hybrid search.
        
        Combines:
        - Vector similarity (semantic)
        - Full-text search (keywords)
        - RRF reranking
        
        Args:
            query: Search query
            top_k: Number of results
            category_filter: Single category to filter
            categories_filter: Multiple categories to filter
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalResult sorted by relevance
        """
        logger.info(
            f"Retriever retrieve with query: {query[:50]}..., "
            f"category_filter: {category_filter}, categories_filter: {categories_filter}"
        )
        
        # Get query embedding
        query_embedding = await self._get_query_embedding(query)
        if query_embedding is None:
            logger.warning("Failed to get embedding, skipping retrieval")
            return []
        
        # Determine categories
        categories = None
        if category_filter:
            categories = [category_filter]
        elif categories_filter:
            categories = categories_filter
        
        # Perform hybrid search
        try:
            results = self.index.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                categories=categories,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}, falling back to vector-only search")
            # Fallback to vector search if hybrid fails
            if categories:
                similarities, indices = self.index.search_with_filter(
                    query_embedding=query_embedding,
                    categories=categories,
                    top_k=top_k,
                )
            else:
                similarities, indices = self.index.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                )
            
            # Convert to result format
            results = []
            table = self.index._get_table()
            df = table.to_pandas()
            for idx, sim in zip(indices, similarities):
                row = df[df["id"] == idx].iloc[0]
                results.append({
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "_distance": 1 - sim,
                    "category": row.get("category", ""),
                    "title": row.get("title", ""),
                    "section": row.get("section", ""),
                    "source_file": row.get("source_file", ""),
                })
        
        # Convert to RetrievalResult
        retrieval_results = []
        for row in results:
            # LanceDB returns _relevance_score for hybrid search
            # or _distance for vector-only search
            if "_relevance_score" in row:
                score = row["_relevance_score"]
            else:
                score = 1 - row.get("_distance", 0)
            
            if score < min_score:
                continue
            
            retrieval_results.append(RetrievalResult(
                chunk_id=row["chunk_id"],
                content=row["content"],
                score=score,
                metadata={
                    "category": row.get("category", ""),
                    "title": row.get("title", ""),
                    "section": row.get("section", ""),
                    "source_file": row.get("source_file", ""),
                },
                retrieval_source="hybrid",
            ))
        
        logger.info(f"Retrieved {len(retrieval_results)} results")
        return retrieval_results
    
    async def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for query."""
        try:
            async with aiohttp.ClientSession() as session:
                embedding = await self.llm_service.get_embedding(
                    session=session,
                    text=query,
                )
                
                if embedding:
                    return np.array(embedding, dtype='float32')
                return None
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None
    
    @classmethod
    def from_directory(
        cls,
        index_dir: str,
        llm_service: LLMService,
    ) -> "LanceDBRetriever":
        """
        Load retriever from index directory.
        
        Args:
            index_dir: Path to directory containing LanceDB database
            llm_service: LLM service for embeddings
            
        Returns:
            Initialized LanceDBRetriever
        """
        lancedb_index = LanceDBIndex.load(index_dir, table_name="knowledge")
        
        logger.info(f"Loaded LanceDB retriever from {index_dir}")
        logger.info(f"  - Vectors: {lancedb_index.ntotal}")
        
        return cls(
            lancedb_index=lancedb_index,
            llm_service=llm_service,
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
    max_chars = max_tokens * 4
    
    for i, result in enumerate(results, 1):
        part = f"[{i}] {result.metadata.get('title', 'Unknown')}\n{result.content}\n"
        
        if total_chars + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_chars += len(part)
    
    return "\n---\n".join(context_parts)

