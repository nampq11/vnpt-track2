"""Build knowledge base index with embeddings."""

import asyncio
import numpy as np
import json
import sys
from pathlib import Path
from typing import List, Optional
import argparse
import aiohttp
from loguru import logger
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.brain.llm.services.vnpt import VNPTService
from src.brain.llm.services.azure import AzureService
from src.brain.llm.services.type import LLMService
from src.brain.rag.document_processor import DocumentProcessor, DocumentChunk, save_chunks
from src.brain.rag.bm25_index import build_bm25_index
from src.brain.rag.faiss_index import build_faiss_index

# Constants
BATCH_SIZE = 50
DELAY_BETWEEN_BATCHES = 0.5  # seconds


async def get_embeddings_batch(
    llm_provider: LLMService,
    session: aiohttp.ClientSession,
    texts: List[str],
) -> List[Optional[List[float]]]:
    """Get embeddings for a batch of texts."""
    tasks = [
        llm_provider.get_embedding(session=session, text=text)
        for text in texts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    embeddings = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to get embedding {i}: {result}")
            embeddings.append(None)
        elif isinstance(result, dict) and 'data' in result:
            # VNPT format
            embeddings.append(result['data'][0].get('embedding'))
        elif isinstance(result, list):
            # Azure format
            embeddings.append(result)
        else:
            logger.warning(f"Unknown embedding format: {type(result)}")
            embeddings.append(None)
    
    return embeddings


async def build_knowledge_index(
    llm_provider: LLMService,
    data_dir: str = "data/data",
    output_dir: str = "data/embeddings/knowledge",
    chunk_size: int = 512,
    overlap: int = 50,
):
    """Main function to build knowledge index."""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Process documents
    logger.info("Processing documents...")
    processor = DocumentProcessor(chunk_size=chunk_size, overlap=overlap)
    chunks = processor.process_directory(Path(data_dir))
    logger.info(f"Created {len(chunks)} chunks from documents")
    
    # Save chunks metadata
    chunks_path = output_path / "chunks.json"
    save_chunks(chunks, str(chunks_path))
    
    # Phase 2: Generate embeddings
    logger.info("Generating embeddings...")
    all_embeddings = []
    valid_indices = []
    
    async with aiohttp.ClientSession() as session:
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding batches"):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c.content for c in batch]
            
            embeddings = await get_embeddings_batch(
                llm_provider=llm_provider,
                session=session,
                texts=texts,
            )
            
            for j, emb in enumerate(embeddings):
                if emb is not None:
                    all_embeddings.append(emb)
                    valid_indices.append(i + j)
            
            # Rate limiting
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)
    
    logger.info(f"Generated {len(all_embeddings)} embeddings ({len(all_embeddings)/len(chunks)*100:.1f}% success rate)")
    
    # Convert to numpy array
    embeddings_matrix = np.array(all_embeddings, dtype='float32')
    dimension = embeddings_matrix.shape[1]
    logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
    
    # Save raw embeddings
    embeddings_path = output_path / "embeddings.npy"
    np.save(str(embeddings_path), embeddings_matrix)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # Build and save FAISS index
    logger.info("Building FAISS index...")
    import faiss
    
    # Normalize for cosine similarity
    embeddings_normalized = embeddings_matrix.copy()
    faiss.normalize_L2(embeddings_normalized)
    
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
    index.add(embeddings_normalized)
    
    faiss_path = output_path / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    
    # Save original embeddings for filtering
    embeddings_orig_path = output_path / "faiss.index.embeddings.npy"
    np.save(str(embeddings_orig_path), embeddings_matrix)
    logger.info(f"Saved FAISS index to {faiss_path}")
    
    # Save valid indices mapping (for handling failed embeddings)
    indices_path = output_path / "valid_indices.json"
    with open(indices_path, "w") as f:
        json.dump(valid_indices, f)
    
    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25_path = output_path / "bm25.pkl"
    build_bm25_index(chunks, str(bm25_path))
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    metadata = {
        "total_chunks": len(chunks),
        "total_embeddings": len(all_embeddings),
        "dimension": dimension,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "provider": llm_provider.__class__.__name__,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✅ Knowledge index build complete!")
    logger.info(f"   - Chunks: {len(chunks)}")
    logger.info(f"   - Embeddings: {len(all_embeddings)}")
    logger.info(f"   - Dimension: {dimension}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build knowledge index with embeddings")
    parser.add_argument(
        "--provider",
        choices=["azure", "vnpt"],
        default="azure",
        help="Embedding provider (default: azure)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/embeddings/knowledge",
        help="Path to output directory"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters (default: 512)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )
    args = parser.parse_args()
    
    # Initialize provider
    if args.provider == "azure":
        logger.info("Using Azure service for embeddings")
        llm_provider = AzureService(
            embedding_model="text-embedding-ada-002"
        )
    else:  # vnpt
        logger.info("Using VNPT service for embeddings")
        llm_provider = VNPTService(
            model_type="embedding"
        )
    
    asyncio.run(build_knowledge_index(
        llm_provider=llm_provider,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    ))

