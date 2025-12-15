"""Safety guardrail using semantic firewall (FR-04)"""

from typing import Optional, List
import numpy as np
from src.core.constants import SAFETY_THRESHOLD
from src.core.models import SafetyCheckResult
from src.runtime.llm.base import LLMService


class SafetyGuard:
    """Semantic firewall for detecting unsafe/harmful questions"""
    
    def __init__(
        self,
        llm_service: LLMService,
        safety_vectors: Optional[np.ndarray] = None,
        threshold: float = SAFETY_THRESHOLD,
    ):
        """
        Initialize safety guard
        
        Args:
            llm_service: LLM service for embedding generation
            safety_vectors: Pre-loaded safety vector matrix (safety.npy)
            threshold: Cosine similarity threshold (default 0.85)
        """
        self.llm_service = llm_service
        self.safety_vectors = safety_vectors
        self.threshold = threshold
    
    async def check(self, query: str) -> SafetyCheckResult:
        """
        Check if query is safe
        
        Args:
            query: User query to check
            
        Returns:
            SafetyCheckResult with is_safe flag and similarity score
        """
        # Get embedding for query
        query_embedding = await self.llm_service.embed(query)
        
        if not query_embedding:
            # If embedding fails, default to safe
            return SafetyCheckResult(is_safe=True, similarity_score=0.0)
        
        # If safety vectors not loaded, can only skip check
        if self.safety_vectors is None:
            return SafetyCheckResult(is_safe=True, similarity_score=0.0)
        
        # Calculate cosine similarity with all safety vectors
        query_vec = np.array(query_embedding).reshape(1, -1)
        
        # Normalize vectors
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        safety_norm = self.safety_vectors / (np.linalg.norm(self.safety_vectors, axis=1, keepdims=True) + 1e-10)
        
        # Compute cosine similarity
        similarities = np.dot(query_norm, safety_norm.T).flatten()
        max_similarity = float(np.max(similarities))
        
        is_safe = max_similarity < self.threshold
        
        return SafetyCheckResult(
            is_safe=is_safe,
            similarity_score=max_similarity,
        )
    
    def load_safety_vectors(self, vectors: np.ndarray) -> None:
        """
        Load safety vector matrix
        
        Args:
            vectors: Numpy array of shape (n_vectors, embedding_dim)
        """
        self.safety_vectors = vectors
    
    def load_from_file(self, path: str) -> None:
        """
        Load safety vectors from .npy file
        
        Args:
            path: Path to safety.npy file
        """
        self.safety_vectors = np.load(path)

