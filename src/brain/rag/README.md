# RAG Module - Hybrid Retrieval System

Hybrid BM25 + FAISS retrieval system with RRF ranking for Vietnamese knowledge base.

## Architecture

```
Query → [BM25 Search] ──┐
                         ├─→ RRF Fusion → Top-k Results
Query → [FAISS Search] ──┘
```

## Components

### 1. Document Processor (`document_processor.py`)
- Parses Vietnamese Wikipedia-style documents
- Extracts metadata (category, title, section)
- Chunks text into 512-char segments with 50-char overlap

### 2. BM25 Index (`bm25_index.py`)
- Lexical search using BM25Okapi algorithm
- Vietnamese tokenization via `underthesea`
- Handles pre-filtering by document indices

### 3. FAISS Index (`faiss_index.py`)
- Vector similarity search using IndexFlatIP
- Cosine similarity (normalized vectors)
- Supports filtering for pre-filtered searches

### 4. Hybrid Retriever (`retriever.py`)
- Combines BM25 + FAISS using RRF (Reciprocal Rank Fusion)
- Pre-filtering by category
- Post-filtering by score threshold

## Usage

### Build Index

```bash
# With Azure embeddings (evaluation)
uv run python -m src.utils.build_knowledge_index --provider azure

# With VNPT embeddings (production)
uv run python -m src.utils.build_knowledge_index --provider vnpt
```

### Use in Code

```python
from src.brain.rag.retriever import HybridRetriever
from src.brain.llm.services.azure import AzureService

# Initialize
llm = AzureService()
retriever = HybridRetriever.from_directory(
    index_dir="data/embeddings/knowledge",
    llm_service=llm,
)

# Search
results = await retriever.retrieve(
    query="Hồ Chí Minh sinh năm nào?",
    top_k=5,
)

for result in results:
    print(f"Score: {result.score:.4f}")
    print(f"Content: {result.content[:100]}...")
```

### Search with Filtering

```python
# Pre-filter by categories
results = await retriever.retrieve(
    query="chiến thắng Điện Biên Phủ",
    top_k=3,
    categories_filter=["Lich_Su_Viet_nam", "khang_chien_lon"],
)

# Post-filter by score
results = await retriever.retrieve(
    query="thủ đô Việt Nam",
    top_k=5,
    min_score=0.01,
)
```

## RRF Algorithm

**Reciprocal Rank Fusion:**
```
RRF_score(d) = Σ weight_i / (k + rank_i(d))
```

- `k` = 60 (default constant)
- `weight_i` = weight for search method i
- `rank_i(d)` = rank of document d in result list i

**Benefits:**
- No score normalization needed
- Handles different score scales
- Robust to outliers

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 512 | Chunk size in characters |
| `overlap` | 50 | Overlap between chunks |
| `rrf_k` | 60 | RRF constant |
| `bm25_weight` | 1.0 | Weight for BM25 scores |
| `faiss_weight` | 1.0 | Weight for FAISS scores |

## Output Files

```
data/embeddings/knowledge/
├── chunks.json              # Chunk metadata
├── embeddings.npy           # Raw embeddings
├── faiss.index             # FAISS index
├── faiss.index.embeddings.npy  # For filtering
├── bm25.pkl                # BM25 index
├── metadata.json           # Build metadata
└── valid_indices.json      # Valid embedding indices
```

## Dependencies

- `faiss-cpu`: Vector search
- `rank-bm25`: BM25 algorithm
- `underthesea`: Vietnamese tokenization

## Testing

```bash
uv run python tests/test_rag_indexing.py
```

