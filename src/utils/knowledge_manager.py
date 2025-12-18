#!/usr/bin/env python3
"""CLI tool for managing LanceDB knowledge base.

Usage:
    # Build index from scratch
    ./bin/knowledge_manager.py build --data-dir data/data --provider azure
    
    # Add/update documents from directory
    ./bin/knowledge_manager.py upsert --data-dir data/data/new_category --provider azure
    
    # Delete documents by file
    ./bin/knowledge_manager.py delete --file "data/data/Bac_Ho/Hồ_Chí_Minh.txt"
    
    # Delete entire category
    ./bin/knowledge_manager.py delete --category "old_category"
    
    # Show index info
    ./bin/knowledge_manager.py info
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
import aiohttp
from loguru import logger

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain.rag.lancedb_index import LanceDBIndex
from src.brain.rag.document_processor import DocumentProcessor, save_chunks, load_chunks
from src.brain.llm.services.azure import AzureService
from src.brain.llm.services.vnpt import VNPTService


class KnowledgeManager:
    """Manager for LanceDB knowledge base operations."""
    
    def __init__(
        self,
        index_dir: str = "data/embeddings/knowledge",
        provider: str = "azure",
    ):
        self.index_dir = Path(index_dir)
        self.provider = provider
        
        # Initialize LLM service
        if provider == "azure":
            self.llm_service = AzureService(embedding_model="text-embedding-ada-002")
            self.dimension = 1536
        else:  # vnpt
            self.llm_service = VNPTService(model_type="embedding")
            self.dimension = 1024
        
        logger.info(f"Initialized KnowledgeManager with {provider} (dim={self.dimension})")
    
    async def build_index(
        self,
        data_dir: str,
        chunk_size: int = 512,
        overlap: int = 50,
        batch_size: int = 50,
    ):
        """Build knowledge index from scratch."""
        logger.info(f"Building index from {data_dir}")
        
        # Create output directory
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Process documents
        logger.info("Processing documents...")
        processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        chunks = processor.process_directory(Path(data_dir))
        logger.info(f"Created {len(chunks)} chunks")
        
        # Save chunks metadata
        chunks_path = self.index_dir / "chunks.json"
        save_chunks(chunks, str(chunks_path))
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = await self._generate_embeddings(
            [c.content for c in chunks],
            batch_size=batch_size
        )
        
        if len(embeddings) != len(chunks):
            logger.error(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")
            return False
        
        # Convert to numpy
        embeddings_matrix = np.array(embeddings, dtype='float32')
        
        # Save embeddings backup
        embeddings_path = self.index_dir / "embeddings.npy"
        np.save(str(embeddings_path), embeddings_matrix)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Build LanceDB index
        logger.info("Building LanceDB index...")
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "category": c.metadata.get("category", "unknown"),
                "title": c.metadata.get("title", ""),
                "section": c.metadata.get("section", ""),
                "source_file": c.metadata.get("source_file", ""),
            }
            for c in chunks
        ]
        
        index = LanceDBIndex(
            db_path=str(self.index_dir),
            table_name="knowledge",
            dimension=self.dimension,
        )
        index.build(embeddings_matrix, chunk_dicts)
        
        # Save metadata
        import json
        metadata = {
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "dimension": self.dimension,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "provider": self.provider,
            "index_type": "lancedb",
        }
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Index build complete!")
        logger.info(f"   - Chunks: {len(chunks)}")
        logger.info(f"   - Embeddings: {len(embeddings)}")
        logger.info(f"   - Location: {self.index_dir}/knowledge.lance/")
        
        return True
    
    async def smart_upsert_documents(
        self,
        data_dir: str,
        chunk_size: int = 512,
        overlap: int = 50,
        batch_size: int = 50,
        skip_indexed: bool = True,
    ):
        """Add or update documents incrementally, skipping already indexed files."""
        logger.info(f"Smart upserting documents from {data_dir}")
        
        # Load existing index
        index = LanceDBIndex.load(str(self.index_dir), table_name="knowledge")
        logger.info(f"Loaded index with {index.ntotal} existing vectors")
        
        # Get already indexed files
        indexed_files = set()
        if skip_indexed:
            indexed_files = index.get_indexed_files()
            logger.info(f"Found {len(indexed_files)} files already indexed")
        
        # Process new documents
        logger.info("Processing documents...")
        processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        all_chunks = processor.process_directory(Path(data_dir))
        
        # Filter out chunks from already indexed files
        if skip_indexed:
            chunks = [
                c for c in all_chunks 
                if c.metadata.get("source_file", "") not in indexed_files
            ]
            skipped = len(all_chunks) - len(chunks)
            logger.info(f"Skipped {skipped} chunks from {len(indexed_files)} indexed files")
        else:
            chunks = all_chunks
        
        logger.info(f"Created {len(chunks)} new chunks to index")
        
        if len(chunks) == 0:
            logger.warning("No new chunks to upsert")
            return False
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = await self._generate_embeddings(
            [c.content for c in chunks],
            batch_size=batch_size
        )
        
        embeddings_matrix = np.array(embeddings, dtype='float32')
        
        # Prepare chunk data
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "category": c.metadata.get("category", "unknown"),
                "title": c.metadata.get("title", ""),
                "section": c.metadata.get("section", ""),
                "source_file": c.metadata.get("source_file", ""),
            }
            for c in chunks
        ]
        
        # Add documents
        logger.info("Adding documents to index...")
        index.add_documents(embeddings_matrix, chunk_dicts)
        
        logger.info("✅ Smart upsert complete!")
        logger.info(f"   - Added: {len(chunks)} chunks")
        logger.info(f"   - Total: {index.ntotal} vectors")
        
        # Update chunks.json
        logger.info("Updating chunks metadata...")
        chunks_path = self.index_dir / "chunks.json"
        if chunks_path.exists():
            existing_chunks = load_chunks(str(chunks_path))
            all_chunks = existing_chunks + chunks
            save_chunks(all_chunks, str(chunks_path))
        
        return True
    
    async def upsert_documents(
        self,
        data_dir: str,
        chunk_size: int = 512,
        overlap: int = 50,
        batch_size: int = 50,
    ):
        """Add or update documents incrementally."""
        logger.info(f"Upserting documents from {data_dir}")
        
        # Load existing index
        index = LanceDBIndex.load(str(self.index_dir), table_name="knowledge")
        logger.info(f"Loaded index with {index.ntotal} existing vectors")
        
        # Process new documents
        logger.info("Processing documents...")
        processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
        chunks = processor.process_directory(Path(data_dir))
        logger.info(f"Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.warning("No chunks to upsert")
            return False
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = await self._generate_embeddings(
            [c.content for c in chunks],
            batch_size=batch_size
        )
        
        embeddings_matrix = np.array(embeddings, dtype='float32')
        
        # Prepare chunk data
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "category": c.metadata.get("category", "unknown"),
                "title": c.metadata.get("title", ""),
                "section": c.metadata.get("section", ""),
                "source_file": c.metadata.get("source_file", ""),
            }
            for c in chunks
        ]
        
        # Add documents
        logger.info("Adding documents to index...")
        index.add_documents(embeddings_matrix, chunk_dicts)
        
        logger.info("✅ Upsert complete!")
        logger.info(f"   - Added: {len(chunks)} chunks")
        logger.info(f"   - Total: {index.ntotal} vectors")
        
        # Update chunks.json
        logger.info("Updating chunks metadata...")
        chunks_path = self.index_dir / "chunks.json"
        if chunks_path.exists():
            existing_chunks = load_chunks(str(chunks_path))
            all_chunks = existing_chunks + chunks
            save_chunks(all_chunks, str(chunks_path))
        
        return True
    
    def delete_by_file(self, source_file: str):
        """Delete documents by source file."""
        logger.info(f"Deleting documents from {source_file}")
        
        # Load index
        index = LanceDBIndex.load(str(self.index_dir), table_name="knowledge")
        logger.info(f"Loaded index with {index.ntotal} vectors")
        
        # Delete
        index.delete_by_source(source_file)
        
        logger.info("✅ Delete complete!")
        logger.info(f"   - Remaining: {index.ntotal} vectors")
        
        return True
    
    def delete_by_category(self, category: str):
        """Delete all documents from a category."""
        logger.info(f"Deleting category: {category}")
        
        # Load index
        index = LanceDBIndex.load(str(self.index_dir), table_name="knowledge")
        logger.info(f"Loaded index with {index.ntotal} vectors")
        
        # Delete using SQL
        table = index._get_table()
        table.delete(f"category = '{category}'")
        
        logger.info("✅ Delete complete!")
        logger.info(f"   - Remaining: {index.ntotal} vectors")
        
        return True
    
    def show_info(self):
        """Show index information."""
        logger.info("Loading index info...")
        
        # Load index
        index = LanceDBIndex.load(str(self.index_dir), table_name="knowledge")
        
        # Get table info
        table = index._get_table()
        df = table.to_pandas()
        
        # Category counts
        category_counts = df["category"].value_counts()
        
        print("\n" + "="*60)
        print("📊 Knowledge Base Information")
        print("="*60)
        print(f"Location: {self.index_dir}/knowledge.lance/")
        print(f"Total vectors: {index.ntotal:,}")
        print(f"Dimension: {index.dimension}")
        print(f"\nCategories ({len(category_counts)}):")
        for cat, count in category_counts.head(20).items():
            print(f"  - {cat}: {count:,} chunks")
        
        if len(category_counts) > 20:
            remaining = len(category_counts) - 20
            print(f"  ... and {remaining} more categories")
        
        # Sample chunks
        print(f"\nSample chunks:")
        for i, row in df.head(3).iterrows():
            print(f"  [{i+1}] {row['chunk_id']}")
            print(f"      Category: {row['category']}")
            print(f"      Title: {row['title']}")
            print(f"      Content: {row['content'][:80]}...")
            print()
        
        print("="*60)
        
        return True
    
    async def _generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 50,
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        from tqdm import tqdm
        
        embeddings = []
        failed_count = 0
        
        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    try:
                        response = await self.llm_service.get_embedding(
                            session=session,
                            text=text,
                        )
                        
                        if isinstance(response, dict) and 'data' in response:
                            # VNPT format
                            emb = response['data'][0].get('embedding')
                        elif isinstance(response, list):
                            # Azure format
                            emb = response
                        else:
                            logger.warning(f"Unknown embedding format: {type(response)}")
                            emb = None
                        
                        if emb:
                            embeddings.append(emb)
                        else:
                            failed_count += 1
                            # Use zero vector as fallback
                            embeddings.append([0.0] * self.dimension)
                    
                    except Exception as e:
                        logger.warning(f"Failed to get embedding: {e}")
                        failed_count += 1
                        embeddings.append([0.0] * self.dimension)
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        if failed_count > 0:
            logger.warning(f"Failed to embed {failed_count}/{len(texts)} texts")
        
        return embeddings


async def main():
    parser = argparse.ArgumentParser(
        description="Manage LanceDB knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build index from scratch")
    build_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing source documents"
    )
    build_parser.add_argument(
        "--index-dir",
        default="data/embeddings/knowledge",
        help="Output directory for index"
    )
    build_parser.add_argument(
        "--provider",
        choices=["azure", "vnpt"],
        default="azure",
        help="Embedding provider"
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters"
    )
    build_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings"
    )
    
    # Upsert command
    upsert_parser = subparsers.add_parser("upsert", help="Add/update documents")
    upsert_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing new documents"
    )
    upsert_parser.add_argument(
        "--index-dir",
        default="data/embeddings/knowledge",
        help="Index directory"
    )
    upsert_parser.add_argument(
        "--provider",
        choices=["azure", "vnpt"],
        default="azure",
        help="Embedding provider"
    )
    upsert_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters"
    )
    upsert_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    upsert_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings"
    )
    
    # Smart Upsert command (skip already indexed files)
    smart_upsert_parser = subparsers.add_parser(
        "smart-upsert", 
        help="Add/update documents, auto-skip already indexed files"
    )
    smart_upsert_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing new documents"
    )
    smart_upsert_parser.add_argument(
        "--index-dir",
        default="data/embeddings/knowledge",
        help="Index directory"
    )
    smart_upsert_parser.add_argument(
        "--provider",
        choices=["azure", "vnpt"],
        default="azure",
        help="Embedding provider"
    )
    smart_upsert_parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters"
    )
    smart_upsert_parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks"
    )
    smart_upsert_parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for embeddings"
    )
    smart_upsert_parser.add_argument(
        "--skip-indexed",
        action="store_true",
        default=True,
        help="Skip already indexed files (default: True)"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete documents")
    delete_parser.add_argument(
        "--file",
        help="Source file path to delete"
    )
    delete_parser.add_argument(
        "--category",
        help="Category to delete"
    )
    delete_parser.add_argument(
        "--index-dir",
        default="data/embeddings/knowledge",
        help="Index directory"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show index information")
    info_parser.add_argument(
        "--index-dir",
        default="data/embeddings/knowledge",
        help="Index directory"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "build":
            manager = KnowledgeManager(
                index_dir=args.index_dir,
                provider=args.provider
            )
            success = await manager.build_index(
                data_dir=args.data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            return 0 if success else 1
        
        elif args.command == "upsert":
            manager = KnowledgeManager(
                index_dir=args.index_dir,
                provider=args.provider
            )
            success = await manager.upsert_documents(
                data_dir=args.data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
            )
            return 0 if success else 1
        
        elif args.command == "smart-upsert":
            manager = KnowledgeManager(
                index_dir=args.index_dir,
                provider=args.provider
            )
            success = await manager.smart_upsert_documents(
                data_dir=args.data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                batch_size=args.batch_size,
                skip_indexed=args.skip_indexed,
            )
            return 0 if success else 1
        
        elif args.command == "delete":
            if not args.file and not args.category:
                logger.error("Must specify either --file or --category")
                return 1
            
            # Use any provider for delete (no embeddings needed)
            manager = KnowledgeManager(
                index_dir=args.index_dir,
                provider="azure"
            )
            
            if args.file:
                success = manager.delete_by_file(args.file)
            else:
                success = manager.delete_by_category(args.category)
            
            return 0 if success else 1
        
        elif args.command == "info":
            manager = KnowledgeManager(
                index_dir=args.index_dir,
                provider="azure"  # No embeddings needed
            )
            success = manager.show_info()
            return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Command failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

