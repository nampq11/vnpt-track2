"""LanceDB vector index for semantic similarity search."""

import numpy as np
import lancedb
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from loguru import logger


class LanceDBIndex:
    """LanceDB vector index for semantic similarity search with hybrid capabilities."""
    
    def __init__(
        self,
        db_path: str,
        table_name: str = "vectors",
        dimension: int = 1536,
    ):
        """
        Initialize LanceDB index.
        
        Args:
            db_path: Path to LanceDB database directory
            table_name: Name of the table
            dimension: Embedding dimension (1536 for Azure, 1024 for VNPT)
        """
        self.db_path = db_path
        self.table_name = table_name
        self.dimension = dimension
        self._db = None
        self._table = None
        
    def _connect(self):
        """Lazy connection to database."""
        if self._db is None:
            self._db = lancedb.connect(self.db_path)
        return self._db
    
    def build(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ):
        """
        Build LanceDB table from embeddings and chunks.
        
        Args:
            embeddings: Numpy array of shape (n_docs, dimension)
            chunks: List of chunk dictionaries with metadata
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )
        
        db = self._connect()
        
        # Prepare data
        data = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            data.append({
                "id": i,
                "chunk_id": chunk.get("chunk_id", str(i)),
                "content": chunk.get("content", ""),
                "vector": emb.tolist(),
                "category": chunk.get("category", "unknown"),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "source_file": chunk.get("source_file", ""),
            })
        
        # Create table (overwrite if exists)
        self._table = db.create_table(
            self.table_name,
            data=data,
            mode="overwrite",
        )
        
        # Create vector index for cosine similarity (skip if too few rows)
        if len(data) >= 256:
            self._table.create_index(metric="cosine")
            logger.info("  - Vector index (cosine similarity)")
        else:
            logger.warning(
                f"  - Skipping vector index (need 256+ rows, have {len(data)})"
            )
        
        # Create full-text search index on content
        self._table.create_fts_index("content")
        logger.info("  - FTS index on content")
        
        # Create scalar indexes for efficient filtering
        self._table.create_scalar_index("chunk_id")
        self._table.create_scalar_index("source_file")
        self._table.create_scalar_index("category")
        logger.info("  - Scalar indexes on chunk_id, source_file, category")
        
        logger.info(
            f"Built LanceDB table with {len(data)} vectors at {self.db_path}"
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using vector similarity.
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            
        Returns:
            Tuple of (similarities, indices) arrays
        """
        table = self._get_table()
        
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(-1)
        
        results = (
            table.search(query)
            .metric("cosine")
            .limit(top_k)
            .to_pandas()
        )
        
        # Extract distances and indices
        distances = results["_distance"].values
        indices = results["id"].values
        
        # Convert distance to similarity (1 - distance for cosine)
        similarities = 1 - distances
        
        return similarities, indices
    
    def search_with_filter(
        self,
        query_embedding: np.ndarray,
        valid_indices: Optional[List[int]] = None,
        categories: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with filtering by indices or categories using SQL WHERE.
        
        Args:
            query_embedding: Query vector
            valid_indices: List of valid document indices
            categories: List of categories to filter
            top_k: Number of results
            
        Returns:
            Tuple of (similarities, indices) arrays
        """
        table = self._get_table()
        
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(-1)
        
        search_query = table.search(query).metric("cosine")
        
        # Apply filters using SQL WHERE clause
        if categories:
            cat_list = ", ".join([f"'{c}'" for c in categories])
            search_query = search_query.where(f"category IN ({cat_list})")
        elif valid_indices:
            idx_list = ", ".join(map(str, valid_indices))
            search_query = search_query.where(f"id IN ({idx_list})")
        
        results = search_query.limit(top_k).to_pandas()
        
        if len(results) == 0:
            return np.array([]), np.array([], dtype=int)
        
        distances = results["_distance"].values
        indices = results["id"].values
        similarities = 1 - distances
        
        return similarities, indices
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        categories: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Native hybrid search combining vector + FTS with RRF reranking.
        
        Based on: https://docs.lancedb.com/search/hybrid-search
        
        Args:
            query_text: Text query for FTS
            query_embedding: Vector query for semantic search
            top_k: Number of results
            categories: Optional category filter
            
        Returns:
            List of result dictionaries
        """
        from lancedb.rerankers import RRFReranker
        
        table = self._get_table()
        
        # Prepare query
        query = query_embedding.astype('float32')
        if query.ndim == 1:
            query = query.reshape(-1).tolist()
        
        # Build hybrid search query
        search_query = (
            table.search(query_type="hybrid")
            .vector(query)
            .text(query_text)
            .limit(top_k * 2)  # Get more for filtering
        )
        
        # Apply category filter if specified
        if categories:
            cat_list = ", ".join([f"'{c}'" for c in categories])
            search_query = search_query.where(f"category IN ({cat_list})")
        
        # Rerank with RRF (Reciprocal Rank Fusion)
        reranker = RRFReranker()
        results = search_query.rerank(reranker).limit(top_k).to_pandas()
        
        return results.to_dict('records')
    
    def add_documents(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ):
        """Add new documents incrementally."""
        table = self._get_table()
        
        # Get current max ID
        current_max_id = table.to_pandas()["id"].max()
        start_id = int(current_max_id) + 1 if not np.isnan(current_max_id) else 0
        
        # Prepare data
        data = []
        for i, (emb, chunk) in enumerate(zip(embeddings, chunks)):
            data.append({
                "id": start_id + i,
                "chunk_id": chunk.get("chunk_id"),
                "content": chunk.get("content", ""),
                "vector": emb.tolist(),
                "category": chunk.get("category", "unknown"),
                "title": chunk.get("title", ""),
                "section": chunk.get("section", ""),
                "source_file": chunk.get("source_file", ""),
            })
        
        # Append to table
        table.add(data)
        
        logger.info(f"Added {len(data)} new documents (IDs {start_id} to {start_id + len(data) - 1})")
    
    def delete_by_source(self, source_file: str):
        """Delete all chunks from a source file."""
        table = self._get_table()
        table.delete(f"source_file = '{source_file}'")
        logger.info(f"Deleted chunks from {source_file}")
    
    def get_indexed_files(self) -> set:
        """Get set of all indexed source files."""
        table = self._get_table()
        df = table.to_pandas()
        return set(df["source_file"].unique())
    
    def _get_table(self):
        """Get or load table."""
        if self._table is None:
            db = self._connect()
            self._table = db.open_table(self.table_name)
        return self._table
    
    @classmethod
    def load(cls, db_path: str, table_name: str = "vectors") -> "LanceDBIndex":
        """Load existing LanceDB index."""
        instance = cls(db_path=db_path, table_name=table_name)
        table = instance._get_table()
        
        # Infer dimension from schema
        schema = table.schema
        vector_field = schema.field("vector")
        # For list type, get the value type
        if hasattr(vector_field.type, "value_type"):
            # Fixed size list
            instance.dimension = vector_field.type.list_size
        
        logger.info(
            f"Loaded LanceDB index from {db_path} with {instance.ntotal} vectors"
        )
        return instance
    
    @property
    def ntotal(self) -> int:
        """Return total number of indexed vectors."""
        try:
            table = self._get_table()
            return table.count_rows()
        except Exception as e:
            logger.debug(f"Could not count rows: {e}")
            return 0

