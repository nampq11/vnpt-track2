# VNPT Track 2 - Titan Shield RAG System

## Purpose
Production-grade Vietnamese multiple-choice question answering system with Retrieval-Augmented Generation (RAG), safety guardrails, and intelligent routing. Built for VNPT Hackathon Track 2.

## Architecture Overview

### Two-Component System

**Component A: Offline Data Pipeline**
- Data crawling with rate limiting
- Vietnamese text cleaning and normalization
- Metadata enrichment using LLM
- Multi-index building: FAISS (vector), BM25 (keyword), Safety (embeddings)
- Unified orchestration for data preparation

**Component B: Docker Runtime Engine**
- Safety guardrails using semantic firewall
- Intelligent question routing (READING/STEM/RAG modes)
- Hybrid search with RRF fusion
- Temporal filtering for time-sensitive documents
- Specialized prompt builders per question type
- VNPT API integration for production inference

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (use `uv run`, `uv add`, `uv sync`)
- **LLM Backends**: 
  - Development: Ollama (local)
  - Production: VNPT API (small/large models)
  - Enrichment: Azure OpenAI
- **Vector Search**: FAISS
- **Keyword Search**: BM25
- **Async Runtime**: aiohttp

## Project Structure
```
src/
├── core/                       # Shared infrastructure
│   ├── config.py              # Unified configuration
│   ├── models.py              # Pydantic data models
│   └── constants.py           # PRD-aligned constants
│
├── offline/                    # Component A: Data Pipeline
│   ├── crawler/               # Data ingestion (rate-limited)
│   ├── processor/             # Text cleaning & enrichment
│   ├── indexer/               # FAISS/BM25/Safety indices
│   └── pipeline.py            # Orchestration
│
├── runtime/                    # Component B: Runtime Engine
│   ├── safety/                # Semantic firewall (FR-04, FR-05)
│   ├── router/                # Question routing (FR-06)
│   ├── rag/                   # Retrieval & fusion (FR-07)
│   ├── llm/                   # LLM & embedding services
│   ├── prompts/               # Mode-specific prompt builders
│   └── pipeline.py            # Main orchestrator
│
└── artifacts/                 # Generated indices
    ├── faiss.index
    ├── bm25.pkl
    ├── safety.npy
    └── metadata.json

data/                          # QA datasets (val.json, test.json)
results/                       # Prediction outputs
```

## VNPT API Integration (Phase 8 - 2024-12-15)

### Chat API
- **Small Model**: `vnptai_hackathon_small` (60 req/h)
  - Endpoint: `/data-service/v1/chat/completions/vnptai-hackathon-small`
- **Large Model**: `vnptai_hackathon_large` (40 req/h)
  - Endpoint: `/data-service/v1/chat/completions/vnptai-hackathon-large`
- **Base URL**: `https://api.idg.vnpt.vn`

### Embedding API
- **Model**: `vnptai_hackathon_embedding`
- **Quota**: 500 req/minute
- **Endpoint**: `/data-service/vnptai-hackathon-embedding`
- **Batch Support**: Yes (with semaphore control)

### Authentication
```bash
export VNPT_API_KEY="<your_bearer_token>"
export VNPT_TOKEN_ID="<your_token_id>"
export VNPT_TOKEN_KEY="<your_token_key>"
export VNPT_MODEL_SIZE="small"  # or "large"
```

## Quick Commands
```bash
# Install dependencies
uv sync --group development

# Quick test (5 questions)
./bin/inference.sh test

# Full evaluation with metrics
./bin/inference.sh eval val

# Inference only (no metrics)
./bin/inference.sh inference test

# Direct CLI with model selection
uv run python predict.py --mode eval --input data/val.json --model qwen3:1.7b

# Use VNPT API (requires credentials)
uv run python predict.py --mode eval --input data/val.json --provider vnpt --model small
```

## Data Format
Questions in `data/*.json`:
- `qid`: Question ID
- `question`: Vietnamese question text (may include context/passage)
- `choices`: Array of 4 options
- `answer`: Letter (A/B/C/D)

## Key Interfaces

### Core Models
- `Chunk` → `src/core/models.py` (text segment with metadata)
- `ChunkMetadata` → `src/core/models.py` (enriched metadata)
- `Question` → `src/core/models.py` (structured question)
- `PredictionResult` → `src/core/models.py` (answer + confidence)

### Offline Pipeline
- `OfflineDataPipeline` → `src/offline/pipeline.py` (main orchestrator)
- `BaseCrawler` → `src/offline/crawler/base.py` (data ingestion)
- `TextCleaner` → `src/offline/processor/cleaner.py` (Vietnamese normalization)
- `Enricher` → `src/offline/processor/enricher.py` (LLM-based enrichment)
- `FAISSBuilder` → `src/offline/indexer/faiss_builder.py` (vector index)
- `BM25Builder` → `src/offline/indexer/bm25_builder.py` (keyword index)
- `SafetyIndexBuilder` → `src/offline/indexer/safety_builder.py` (safety embeddings)

### Runtime Engine
- `RuntimeInferencePipeline` → `src/runtime/pipeline.py` (main entry)
- `SafetyGuard` → `src/runtime/safety/guard.py` (semantic firewall)
- `SafetySelector` → `src/runtime/safety/selector.py` (safety answer filter)
- `RegexRouter` → `src/runtime/router/regex_router.py` (question classification)
- `HybridSearchEngine` → `src/runtime/rag/hybrid_search.py` (retrieval)
- `TemporalFilter` → `src/runtime/rag/temporal_filter.py` (time-based filtering)
- `ReciprocalRankFusion` → `src/runtime/rag/fusion.py` (result fusion)

### LLM Services
- `VNPTService` → `src/runtime/llm/vnpt_service.py` (Chat API)
- `VNPTEmbeddingService` → `src/runtime/llm/embedding.py` (Embedding API)
- `OllamaService` → `src/brain/llm/services/ollama.py` (Development)
- `AzureService` → `src/brain/llm/services/azure.py` (Enrichment)

### Configuration
- `Config` → `src/core/config.py` (unified configuration)
- `VNPTConfig` → `src/core/config.py` (VNPT API credentials)
- `OllamaConfig` → `src/core/config.py` (development setup)
- `RuntimeConfig` → `src/core/config.py` (runtime parameters)

## Conventions
- **Async-first**: Use `async/await` for all LLM and API calls
- **Type hints**: Required on all public functions and methods
- **Vietnamese support**: UTF-8 encoding throughout, proper text normalization
- **Error handling**: Comprehensive with detailed messages
- **Modular design**: Clear separation of concerns across components
- **Validation**: Configuration validation at startup

## Documentation
- `VNPT_API_INTEGRATION.md` - Integration guide and API reference
- `CHANGELOG_VNPT_INTEGRATION.md` - Detailed change log for Phase 8
- `context_for_ai/tasks/vnpt_api_integration.md` - Task documentation
- `context_for_ai/progress.md` - Overall progress tracking
- `context_for_ai/prd.md` - Product requirements
- `context_for_ai/llm_api_description.md` - VNPT API specification

## Development vs Production

### Development (Ollama)
```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434/v1
export OLLAMA_MODEL=qwen3:1.7b
```

### Production (VNPT)
```bash
export LLM_PROVIDER=vnpt
export VNPT_API_KEY=<bearer_token>
export VNPT_TOKEN_ID=<token_id>
export VNPT_TOKEN_KEY=<token_key>
export VNPT_MODEL_SIZE=small
```

## Important Notes
- Do not create new documentation files; update existing files
- All code changes must be properly tested
- Type hints and comprehensive docstrings required
- UTF-8 encoding for Vietnamese text handling
- Async patterns throughout for scalability
