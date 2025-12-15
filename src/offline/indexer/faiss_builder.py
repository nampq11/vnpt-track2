"""FAISS vector index builder"""

from typing import List, Tuple
import numpy as np
from src.core.models import Chunk
from src.runtime.llm.base import LLMService


class FAISSIndexBuilder:
    """Builds FAISS index from chunks"""
    
    def __init__(self, llm_service: LLMService):
        """
        Initialize FAISS index builder
        
        Args:
            llm_service: LLM service for generating embeddings
        """
        self.llm_service = llm_service
        self.index = None
        self.embeddings = None
    
    async def build(
        self,
        chunks: List[Chunk],
        embedding_dim: int = 1024,
    ) -> Tuple:
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunks to index
            embedding_dim: Dimension of embeddings
            
        Returns:
            Tuple of (faiss_index, embeddings_array)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embeddings = await self._generate_embeddings(texts)
        
        if not embeddings:
            raise ValueError("Failed to generate embeddings")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Verify dimension matches
        if embeddings_array.shape[1] != embedding_dim and embedding_dim != -1:
            print(f"Warning: Embedding dimension {embeddings_array.shape[1]} doesn't match {embedding_dim}")
        
        # Create and train index
        # Using IndexFlatL2 for simplicity (no training required)
        # For production, could use IndexIVFFlat or IndexHNSW
        index = faiss.IndexFlatL2(embeddings_array.shape[1])
        index.add(embeddings_array)
        
        self.index = index
        self.embeddings = embeddings_array
        
        return index, embeddings_array
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            try:
                embedding = await self.llm_service.embed(text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Embedding error: {str(e)}")
                embeddings.append([])
        
        return embeddings
    
    def save(self, path: str) -> None:
        """
        Save FAISS index to file
        
        Args:
            path: Output file path
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        try:
            import faiss
            faiss.write_index(self.index, path)
            print(f"FAISS index saved to {path}")
        except Exception as e:
            print(f"Failed to save FAISS index: {str(e)}")
    
    def save_embeddings(self, path: str) -> None:
        """
        Save embeddings to .npy file
        
        Args:
            path: Output file path
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call build() first.")
        
        np.save(path, self.embeddings)
        print(f"Embeddings saved to {path}")

