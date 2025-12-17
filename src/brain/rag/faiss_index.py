"""FAISS vector index for semantic similarity search."""

import numpy as np
import faiss
from typing import Tuple, Optional, List
from pathlib import Path
from loguru import logger


class FAISSIndex:
    """FAISS vector index for semantic similarity search."""
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension (1536 for Azure, 1024 for VNPT)
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatIP] = None
        self._embeddings: Optional[np.ndarray] = None
        
    def build(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of shape (n_docs, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )
        
        # Store original embeddings for potential reuse
        self._embeddings = embeddings.copy()
        
        # Normalize embeddings for cosine similarity
        # IndexFlatIP computes inner product; normalized vectors = cosine similarity
        embeddings_normalized = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Create flat index (exact search, good for < 1M vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_normalized)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build() first.")
        
        # Ensure proper shape
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Validate dimension
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query.shape[1]}"
            )
        
        # Normalize query for cosine similarity
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, top_k)
        
        return distances[0], indices[0]  # Return 1D arrays
    
    def search_batch(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            query_embeddings: Query vectors of shape (n_queries, dimension)
            top_k: Number of results per query
            
        Returns:
            Tuple of (distances, indices) arrays of shape (n_queries, top_k)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build() first.")
        
        queries = query_embeddings.astype('float32')
        faiss.normalize_L2(queries)
        
        return self.index.search(queries, top_k)
    
    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        valid_indices: List[int],
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with pre-filtering by valid indices.
        
        This is less efficient than native FAISS filtering but works
        for small filtered sets.
        
        Args:
            query_embedding: Query vector
            valid_indices: List of valid document indices
            top_k: Number of results
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if self._embeddings is None:
            raise RuntimeError("Original embeddings not available for filtering")
        
        # Get embeddings for valid indices only
        filtered_embeddings = self._embeddings[valid_indices]
        
        # Build temporary index
        temp_index = faiss.IndexFlatIP(self.dimension)
        filtered_normalized = filtered_embeddings.astype('float32')
        faiss.normalize_L2(filtered_normalized)
        temp_index.add(filtered_normalized)
        
        # Search
        query = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)
        distances, local_indices = temp_index.search(query, min(top_k, len(valid_indices)))
        
        # Map back to original indices
        global_indices = np.array([valid_indices[i] for i in local_indices[0]])
        
        return distances[0], global_indices
    
    def save(self, path: str):
        """Save FAISS index to file."""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        faiss.write_index(self.index, path)
        logger.info(f"Saved FAISS index to {path}")
        
        # Also save embeddings for filtering support
        embeddings_path = Path(path).with_suffix('.embeddings.npy')
        if self._embeddings is not None:
            np.save(str(embeddings_path), self._embeddings)
    
    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """Load FAISS index from file."""
        index = faiss.read_index(path)
        
        instance = cls(dimension=index.d)
        instance.index = index
        
        # Try to load embeddings (must match save path pattern)
        embeddings_path = Path(path).parent / f"{Path(path).name}.embeddings.npy"
        if embeddings_path.exists():
            instance._embeddings = np.load(str(embeddings_path))
            logger.debug(f"Loaded embeddings with shape {instance._embeddings.shape}")
        else:
            logger.warning(f"Embeddings file not found at {embeddings_path}")
        
        logger.info(f"Loaded FAISS index from {path} with {index.ntotal} vectors")
        return instance
    
    @property
    def ntotal(self) -> int:
        """Return total number of indexed vectors."""
        return self.index.ntotal if self.index else 0


def build_faiss_index(
    embeddings: np.ndarray,
    output_path: str,
) -> FAISSIndex:
    """Build and save FAISS index."""
    dimension = embeddings.shape[1]
    index = FAISSIndex(dimension=dimension)
    index.build(embeddings)
    index.save(output_path)
    return index

