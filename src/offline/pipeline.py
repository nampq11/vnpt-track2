"""Offline data pipeline orchestration"""

import json
import asyncio
from typing import List, Optional
from pathlib import Path

from src.core.models import Chunk, ChunkMetadata, ChunkType
from src.core.config import Config
from src.offline.processor.cleaner import TextCleaner
from src.offline.processor.enricher import Enricher
from src.offline.indexer.faiss_builder import FAISSIndexBuilder
from src.offline.indexer.bm25_builder import BM25IndexBuilder
from src.offline.indexer.safety_builder import SafetyIndexBuilder
from src.runtime.llm.base import LLMService


class OfflineDataPipeline:
    """Orchestrates the complete offline data processing pipeline"""
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[Config] = None,
    ):
        """
        Initialize offline pipeline
        
        Args:
            llm_service: LLM service (Azure/GPT-4 for enrichment)
            config: Configuration object
        """
        self.llm_service = llm_service
        self.config = config or Config()
        
        self.cleaner = TextCleaner()
        self.enricher = Enricher(llm_service)
        
        self.chunks: List[Chunk] = []
        
        # Create artifacts directory
        Path(self.config.offline.output_dir).mkdir(parents=True, exist_ok=True)
    
    async def process_documents(
        self,
        raw_documents: List[str],
        enable_enrichment: bool = True,
    ) -> List[Chunk]:
        """
        Process raw documents into chunks with metadata
        
        Args:
            raw_documents: List of raw document texts
            enable_enrichment: Whether to use LLM for enrichment
            
        Returns:
            List of processed Chunk objects
        """
        chunks = []
        chunk_id_counter = 0
        
        for doc_idx, doc_text in enumerate(raw_documents):
            # Clean text
            cleaned = self.cleaner.clean(doc_text)
            
            # Split into chunks
            text_chunks = self.cleaner.chunk_text(
                cleaned,
                chunk_size=self.config.offline.chunk_size,
                overlap=self.config.offline.chunk_overlap,
            )
            
            # Process each chunk
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk_id = f"chunk_{doc_idx:05d}_{chunk_idx:03d}"
                
                # Enrich with metadata if enabled
                if enable_enrichment:
                    metadata = await self.enricher.enrich_chunk(chunk_id, chunk_text)
                else:
                    # Default metadata
                    metadata = ChunkMetadata(
                        source=f"doc_{doc_idx}",
                        type=ChunkType.GENERAL,
                        valid_from=1900,
                        expire_at=9999,
                        province="ALL",
                    )
                
                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=metadata,
                )
                
                chunks.append(chunk)
                chunk_id_counter += 1
        
        self.chunks = chunks
        print(f"Processed {len(chunks)} chunks from {len(raw_documents)} documents")
        
        return chunks
    
    async def build_indices(self) -> dict:
        """
        Build all indices: FAISS, BM25, and Safety
        
        Returns:
            Dictionary with paths to built indices
        """
        if not self.chunks:
            raise ValueError("No chunks processed. Call process_documents() first.")
        
        results = {}
        
        # Build FAISS index
        print("Building FAISS vector index...")
        faiss_builder = FAISSIndexBuilder(self.llm_service)
        faiss_index, embeddings = await faiss_builder.build(self.chunks)
        faiss_path = f"{self.config.offline.output_dir}/faiss.index"
        faiss_builder.save(faiss_path)
        results['faiss'] = faiss_path
        
        # Build BM25 index
        print("Building BM25 keyword index...")
        bm25_builder = BM25IndexBuilder()
        bm25_index = bm25_builder.build(self.chunks)
        bm25_path = f"{self.config.offline.output_dir}/bm25.pkl"
        bm25_builder.save(bm25_path)
        results['bm25'] = bm25_path
        
        # Build Safety index
        print("Building Safety vector index...")
        safety_builder = SafetyIndexBuilder(self.llm_service)
        harmful_questions = safety_builder.generate_synthetic_questions()
        safety_vectors = await safety_builder.build(harmful_questions)
        safety_path = f"{self.config.offline.output_dir}/safety.npy"
        safety_builder.save(safety_path)
        results['safety'] = safety_path
        
        return results
    
    def save_metadata(self) -> str:
        """
        Save chunk metadata to JSON
        
        Returns:
            Path to saved metadata file
        """
        metadata_list = []
        
        for chunk in self.chunks:
            metadata_list.append({
                "id": chunk.id,
                "source": chunk.metadata.source,
                "type": chunk.metadata.type.value,
                "valid_from": chunk.metadata.valid_from,
                "expire_at": chunk.metadata.expire_at,
                "province": chunk.metadata.province,
            })
        
        metadata_path = f"{self.config.offline.output_dir}/metadata.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
        
        return metadata_path
    
    async def run(
        self,
        raw_documents: List[str],
    ) -> dict:
        """
        Run complete offline pipeline
        
        Args:
            raw_documents: List of raw document texts
            
        Returns:
            Dictionary with paths to all generated artifacts
        """
        print("=" * 60)
        print("OFFLINE DATA PIPELINE")
        print("=" * 60)
        
        # Process documents
        print("\nStep 1: Processing documents...")
        await self.process_documents(
            raw_documents,
            enable_enrichment=self.config.offline.enable_enrichment,
        )
        
        # Build indices
        print("\nStep 2: Building indices...")
        indices = await self.build_indices()
        
        # Save metadata
        print("\nStep 3: Saving metadata...")
        self.save_metadata()
        
        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total chunks: {len(self.chunks)}")
        print(f"FAISS index: {indices['faiss']}")
        print(f"BM25 index: {indices['bm25']}")
        print(f"Safety index: {indices['safety']}")
        print(f"Metadata: {self.config.offline.output_dir}/metadata.json")
        print("=" * 60)
        
        return {
            **indices,
            'metadata': f"{self.config.offline.output_dir}/metadata.json",
            'chunks_count': len(self.chunks),
        }

