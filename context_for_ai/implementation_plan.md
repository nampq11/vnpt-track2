# Implementation Plan - Critical Fixes & Improvements

**Date**: December 15, 2024  
**Based On**: Codebase Review Report  
**Priority**: Execute P0 tasks immediately

---

## PHASE 1: CRITICAL BLOCKERS (8-12 hours)

### Task 1.1: Implement BM25 Search ⏱️ 2-4 hours
**Priority**: P0 (BLOCKER)  
**File**: `src/runtime/rag/hybrid_search.py`  
**Line**: 96-122

**Current Issue**:
```python
async def _bm25_search(self, query: str, top_k: int):
    # TODO: Implement bm25s search
    return []  # Always empty!
```

**Implementation Steps**:

1. Load BM25 index properly in `__init__`:
```python
def __init__(self, llm_service: LLMService):
    # Add BM25 initialization
    import bm25s
    self.bm25_model = None
    self.bm25_corpus_tokens = None
```

2. Implement index loading:
```python
def load_bm25_index(self, index_path: str, metadata_path: str):
    """Load BM25 index from file"""
    import bm25s
    import json
    
    # Load BM25 model
    self.bm25_model = bm25s.BM25.load(index_path)
    
    # Load tokenized corpus (needed for retrieval)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        # Reconstruct corpus from metadata
```

3. Implement search:
```python
async def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
    if self.bm25_model is None:
        return []
    
    try:
        # Tokenize query
        import bm25s.tokenization
        tokenizer = bm25s.tokenization.Tokenizer()
        query_tokens = tokenizer.tokenize([query], return_ids=True)
        
        # Search
        results, scores = self.bm25_model.retrieve(
            query_tokens, 
            k=top_k
        )
        
        # Map to chunks
        chunk_results = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self.chunks):
                chunk_results.append((self.chunks[idx], float(score)))
        
        return chunk_results
    except Exception as e:
        print(f"BM25 search error: {str(e)}")
        return []
```

4. Update RuntimeInferencePipeline to call load methods:
```python
def _load_artifacts(self) -> None:
    try:
        # Load FAISS
        faiss_path = Path(self.config.runtime.artifacts_dir) / "faiss.index"
        if faiss_path.exists():
            self.rag_engine.load_faiss_index(str(faiss_path))
        
        # Load BM25
        bm25_path = Path(self.config.runtime.artifacts_dir) / "bm25.pkl"
        metadata_path = Path(self.config.runtime.artifacts_dir) / "metadata.json"
        if bm25_path.exists() and metadata_path.exists():
            self.rag_engine.load_bm25_index(str(bm25_path), str(metadata_path))
        
        # Load safety
        safety_path = Path(self.config.runtime.artifacts_dir) / "safety.npy"
        if safety_path.exists():
            self.safety_guard.load_from_file(str(safety_path))
            
    except Exception as e:
        print(f"Warning: Could not load artifacts: {str(e)}")
```

**Testing**:
```bash
# After implementation
uv run pytest tests/test_runtime_rag.py::test_bm25_search -v
```

---

### Task 1.2: Fix Search Result Conversion ⏱️ 1-2 hours
**Priority**: P0 (BLOCKER)  
**File**: `src/runtime/pipeline.py`  
**Line**: 155-158

**Current Issue**:
```python
async def _get_search_results(self, route):
    """Helper to get search results from route"""
    return None  # Always None!
```

**Implementation**:
```python
async def _get_search_results(self, route):
    """Convert route chunks to SearchResult objects for prompt building"""
    if not route.context_chunks:
        return []
    
    from src.runtime.rag.hybrid_search import SearchResult
    
    # Convert chunks to SearchResult format
    results = []
    for i, chunk in enumerate(route.context_chunks):
        # Use decreasing scores for ranking (if route doesn't have scores)
        score = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, etc.
        results.append(SearchResult(
            chunk=chunk,
            score=max(score, 0.1),  # Minimum 0.1
            source="hybrid"
        ))
    
    return results
```

**Alternative** (if route stores scores):
```python
async def _get_search_results(self, route):
    """Convert route chunks with scores to SearchResult objects"""
    if not hasattr(route, 'context_chunks') or not route.context_chunks:
        return []
    
    from src.runtime.rag.hybrid_search import SearchResult
    
    results = []
    scores = getattr(route, 'context_scores', [1.0] * len(route.context_chunks))
    
    for chunk, score in zip(route.context_chunks, scores):
        results.append(SearchResult(
            chunk=chunk,
            score=float(score),
            source=getattr(route, 'source', 'hybrid')
        ))
    
    return results
```

---

### Task 1.3: Generate Sample Artifacts ⏱️ 4-6 hours
**Priority**: P0 (BLOCKER)  
**File**: Create `scripts/generate_sample_artifacts.py`

**Purpose**: Create minimal but functional indices for testing

**Implementation**:

```python
#!/usr/bin/env python3
"""Generate sample artifacts for testing runtime pipeline"""

import asyncio
import json
from pathlib import Path
import numpy as np

from src.core.models import Chunk, ChunkMetadata, ChunkType
from src.core.config import Config
from src.offline.indexer.faiss_builder import FAISSIndexBuilder
from src.offline.indexer.bm25_builder import BM25IndexBuilder
from src.offline.indexer.safety_builder import SafetyIndexBuilder
from src.brain.llm.services.ollama import OllamaService


async def main():
    print("=" * 60)
    print("GENERATING SAMPLE ARTIFACTS")
    print("=" * 60)
    
    # Initialize
    config = Config.from_env()
    output_dir = Path(config.offline.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load knowledge base
    kb_file = Path("data/knowledge_base.json")
    if not kb_file.exists():
        print("⚠️  knowledge_base.json not found, creating sample data...")
        sample_data = create_sample_knowledge()
    else:
        with open(kb_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    
    # Create chunks
    print(f"\n1. Creating chunks from {len(sample_data)} documents...")
    chunks = create_chunks_from_knowledge(sample_data)
    print(f"   Created {len(chunks)} chunks")
    
    # Initialize LLM service for embedding
    print("\n2. Initializing LLM service...")
    llm_service = OllamaService(model="qwen3:1.7b")
    
    # Build FAISS index
    print("\n3. Building FAISS index...")
    faiss_builder = FAISSIndexBuilder(llm_service)
    faiss_index, embeddings = await faiss_builder.build(chunks)
    faiss_path = output_dir / "faiss.index"
    faiss_builder.save(str(faiss_path))
    print(f"   ✓ Saved to {faiss_path}")
    
    # Build BM25 index
    print("\n4. Building BM25 index...")
    bm25_builder = BM25IndexBuilder()
    bm25_index = bm25_builder.build(chunks)
    bm25_path = output_dir / "bm25.pkl"
    bm25_builder.save(str(bm25_path))
    print(f"   ✓ Saved to {bm25_path}")
    
    # Build Safety index
    print("\n5. Building Safety index...")
    safety_builder = SafetyIndexBuilder(llm_service)
    harmful_questions = safety_builder.generate_synthetic_questions()
    safety_vectors = await safety_builder.build(harmful_questions)
    safety_path = output_dir / "safety.npy"
    safety_builder.save(str(safety_path))
    print(f"   ✓ Saved to {safety_path}")
    print(f"   Generated {len(harmful_questions)} harmful question vectors")
    
    # Save metadata
    print("\n6. Saving metadata...")
    save_metadata(chunks, output_dir / "metadata.json")
    print(f"   ✓ Saved chunk metadata")
    
    print("\n" + "=" * 60)
    print("ARTIFACTS GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"FAISS vectors: {len(embeddings)}")
    print(f"Output directory: {output_dir}")
    print("\nYou can now run runtime tests:")
    print("  uv run pytest tests/test_runtime_rag.py -v")


def create_sample_knowledge():
    """Create sample knowledge base if file doesn't exist"""
    return [
        {
            "title": "Luật Đất đai 2024",
            "content": "Luật Đất đai năm 2024 có hiệu lực từ ngày 1 tháng 1 năm 2025. "
                      "Luật này quy định về chế độ sở hữu, quyền sử dụng đất, "
                      "nghĩa vụ và trách nhiệm của người sử dụng đất.",
            "year": 2024,
            "type": "LAW"
        },
        {
            "title": "Hiến pháp 2013",
            "content": "Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam năm 2013. "
                      "Điều 1: Nước Cộng hòa xã hội chủ nghĩa Việt Nam là một nước "
                      "độc lập, có chủ quyền, thống nhất và toàn vẹn lãnh thổ.",
            "year": 2013,
            "type": "LAW"
        },
        {
            "title": "Toán học - Đạo hàm",
            "content": "Đạo hàm của hàm số f(x) = x^2 là f'(x) = 2x. "
                      "Đạo hàm là giới hạn của tỉ số giữa số gia của hàm số và số gia "
                      "của biến số khi số gia của biến số tiến tới 0.",
            "year": 2020,
            "type": "MATH"
        },
        {
            "title": "Lịch sử Việt Nam",
            "content": "Cách mạng tháng Tám năm 1945 là cuộc cách mạng giải phóng dân tộc "
                      "của nhân dân Việt Nam do Đảng Cộng sản Đông Dương và Chủ tịch Hồ Chí Minh lãnh đạo.",
            "year": 1945,
            "type": "HISTORY"
        },
    ]


def create_chunks_from_knowledge(knowledge_data):
    """Convert knowledge base to chunks"""
    chunks = []
    
    for idx, doc in enumerate(knowledge_data):
        chunk_id = f"chunk_{idx:05d}"
        
        metadata = ChunkMetadata(
            source=doc.get("title", f"doc_{idx}"),
            type=ChunkType[doc.get("type", "GENERAL")],
            valid_from=doc.get("year", 2020),
            expire_at=9999,
            province="ALL"
        )
        
        chunk = Chunk(
            id=chunk_id,
            text=doc.get("content", ""),
            metadata=metadata
        )
        
        chunks.append(chunk)
    
    return chunks


def save_metadata(chunks, output_path):
    """Save chunk metadata to JSON"""
    metadata_list = []
    
    for chunk in chunks:
        metadata_list.append({
            "id": chunk.id,
            "text": chunk.text[:100],  # Preview
            "source": chunk.metadata.source,
            "type": chunk.metadata.type.value,
            "valid_from": chunk.metadata.valid_from,
            "expire_at": chunk.metadata.expire_at,
            "province": chunk.metadata.province,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
```

**Usage**:
```bash
# Generate artifacts
uv run python scripts/generate_sample_artifacts.py

# Verify creation
ls -lh src/artifacts/
# Should show: faiss.index, bm25.pkl, safety.npy, metadata.json
```

---

## PHASE 2: RUNTIME TESTS (3-4 hours)

### Task 2.1: Safety Component Tests
**File**: Create `tests/test_runtime_safety.py`

```python
"""Tests for runtime safety components"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.runtime.safety.guard import SafetyGuard
from src.runtime.safety.selector import SafetySelector
from src.core.models import Question
from src.runtime.llm.base import LLMServiceResponse


class TestSafetyGuard:
    """Test safety detection"""
    
    @pytest.fixture
    def mock_llm_service(self):
        service = Mock()
        service.embed = AsyncMock(return_value=[0.1] * 768)
        return service
    
    @pytest.fixture
    def safety_guard(self, mock_llm_service):
        guard = SafetyGuard(mock_llm_service)
        # Mock safety vectors
        import numpy as np
        guard.safety_vectors = np.random.rand(10, 768)
        return guard
    
    @pytest.mark.asyncio
    async def test_safe_question(self, safety_guard):
        """Test that safe questions pass"""
        result = await safety_guard.check("What is the capital of Vietnam?")
        assert result.is_safe is True
        assert result.max_similarity < 0.85
    
    @pytest.mark.asyncio
    async def test_harmful_question(self, safety_guard, mock_llm_service):
        """Test that harmful questions are detected"""
        # Mock high similarity
        mock_llm_service.embed.return_value = guard.safety_vectors[0].tolist()
        
        result = await safety_guard.check("How to fake a stamp?")
        assert result.is_safe is False
        assert result.max_similarity >= 0.85


class TestSafetySelector:
    """Test safety answer selection"""
    
    @pytest.fixture
    def mock_llm_service(self):
        service = Mock()
        service.generate = AsyncMock(
            return_value=LLMServiceResponse(content="Answer: B", usage={})
        )
        return service
    
    @pytest.fixture
    def safety_selector(self, mock_llm_service):
        return SafetySelector(mock_llm_service)
    
    @pytest.mark.asyncio
    async def test_finds_refusal_keyword(self, safety_selector):
        """Test regex detection of refusal keywords"""
        question = Question(
            qid="test1",
            question="Is this legal?",
            choices=[
                "Yes, it's fine",
                "No, this is illegal and prohibited",
                "Maybe",
                "I don't know"
            ],
            answer="B"
        )
        
        answer = await safety_selector.select_answer(question)
        assert answer == "B"
    
    @pytest.mark.asyncio
    async def test_llm_fallback(self, safety_selector, mock_llm_service):
        """Test LLM fallback when regex fails"""
        question = Question(
            qid="test2",
            question="What should I do?",
            choices=[
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            answer="A"
        )
        
        answer = await safety_selector.select_answer(question)
        assert answer in ["A", "B", "C", "D"]
        assert mock_llm_service.generate.called
```

### Task 2.2: Router Tests
**File**: Create `tests/test_runtime_router.py`

```python
"""Tests for runtime router"""
import pytest
from src.runtime.router.regex_router import RegexRouter
from src.core.models import Question
from src.core.constants import RouteMode


class TestRegexRouter:
    """Test question routing logic"""
    
    @pytest.fixture
    def router(self):
        return RegexRouter()
    
    def test_reading_mode_detection(self, router):
        """Test READING mode for passage-based questions"""
        question = Question(
            qid="q1",
            question="Đoạn văn sau: 'This is a passage...'. Question?",
            choices=["A", "B", "C", "D"]
        )
        
        route = router.route_with_context(question)
        assert route.mode == RouteMode.READING
    
    def test_stem_mode_detection(self, router):
        """Test STEM mode for math questions"""
        question = Question(
            qid="q2",
            question="Tính đạo hàm của hàm số f(x) = x^2",
            choices=["2x", "x", "2", "x^2"]
        )
        
        route = router.route_with_context(question)
        assert route.mode == RouteMode.STEM
    
    def test_rag_mode_default(self, router):
        """Test RAG mode as default"""
        question = Question(
            qid="q3",
            question="Luật Đất đai 2024 có hiệu lực từ khi nào?",
            choices=["2024", "2025", "2026", "2027"]
        )
        
        route = router.route_with_context(question)
        assert route.mode == RouteMode.RAG
```

---

## PHASE 3: CONSOLIDATION (1-2 hours)

### Task 3.1: Mark Legacy Code as Deprecated
**Files**: `src/brain/inference/`, `src/brain/config.py`

**Actions**:
1. Add deprecation warning to brain/inference/__init__.py:
```python
"""
DEPRECATED: This module is deprecated as of v2.0.0

Use src.runtime.pipeline.RuntimeInferencePipeline instead.
This module is kept for backward compatibility only.
"""
import warnings
warnings.warn(
    "src.brain.inference is deprecated. Use src.runtime.pipeline instead.",
    DeprecationWarning,
    stacklevel=2
)
```

2. Update tests to use runtime components:
```python
# OLD:
from src.brain.inference.processor import Question

# NEW:
from src.core.models import Question
```

3. Update README.md to clarify:
```markdown
## Architecture (v2.0+)

**Primary Pipeline**: `src/runtime/` (Production)
- RuntimeInferencePipeline
- Safety, Router, RAG components

**Legacy Pipeline**: `src/brain/` (Deprecated)
- Kept for backward compatibility
- Will be removed in v3.0
```

---

## PHASE 4: ERROR RECOVERY (2-3 hours)

### Task 4.1: Add Retry Logic
**File**: Create `src/runtime/llm/retry.py`

```python
"""Retry logic with exponential backoff"""
import asyncio
from typing import Callable, TypeVar, Any
from functools import wraps

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    Retry async function with exponential backoff
    
    Args:
        func: Async function to retry
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = min(base_delay * (2 ** attempt), max_delay)
                print(f"Retry {attempt + 1}/{max_retries} after {delay}s: {str(e)}")
                await asyncio.sleep(delay)
            else:
                print(f"All retries failed: {str(e)}")
    
    raise last_exception


def with_retry(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry logic"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                lambda: func(*args, **kwargs),
                max_retries=max_retries,
                base_delay=base_delay
            )
        return wrapper
    return decorator
```

### Task 4.2: Apply Retry to VNPT Service
**File**: `src/runtime/llm/vnpt_service.py`

```python
from .retry import with_retry

class VNPTService(LLMService):
    @with_retry(max_retries=3, base_delay=1.0)
    async def generate(self, user_input: str, **kwargs) -> LLMServiceResponse:
        # Existing implementation...
        pass
    
    @with_retry(max_retries=3, base_delay=1.0)
    async def embed(self, text: str, **kwargs) -> List[float]:
        # Existing implementation...
        pass
```

---

## EXECUTION TIMELINE

### Day 1 (8-10 hours)
- ✅ Morning: Task 1.1 - BM25 Implementation (2-4h)
- ✅ Afternoon: Task 1.2 - Search Result Fix (1-2h)
- ✅ Evening: Task 1.3 - Generate Artifacts (4-6h)

**Milestone**: RAG mode functional

### Day 2 (4-6 hours)
- ✅ Morning: Task 2.1 - Safety Tests (1.5-2h)
- ✅ Morning: Task 2.2 - Router Tests (1.5-2h)
- ✅ Afternoon: Task 3.1 - Deprecate Legacy (1-2h)

**Milestone**: Test coverage >50%

### Day 3 (2-3 hours)
- ✅ Morning: Task 4.1 - Retry Logic (1-1.5h)
- ✅ Afternoon: Task 4.2 - Apply Retries (1-1.5h)

**Milestone**: Production-ready

---

## SUCCESS CRITERIA

**Phase 1 Complete**:
- [ ] BM25 search returns results
- [ ] RAG mode retrieves context
- [ ] All tests pass: `pytest tests/ -v`

**Phase 2 Complete**:
- [ ] Safety tests pass
- [ ] Router tests pass
- [ ] Test coverage >50%

**Phase 3 Complete**:
- [ ] Legacy code marked deprecated
- [ ] Documentation updated
- [ ] No confusion about which pipeline to use

**Phase 4 Complete**:
- [ ] Retry logic added
- [ ] API errors handled gracefully
- [ ] System degrades gracefully on failures

---

## ROLLBACK PLAN

If issues arise:

1. **BM25 fails**: Fall back to vector-only search
2. **Tests fail**: Fix tests, not implementation
3. **Artifacts generation fails**: Use minimal mock data
4. **API errors**: Implement circuit breaker (skip retry for now)

---

**Plan Created**: December 15, 2024  
**Estimated Completion**: 3 days  
**Risk Level**: Low (incremental changes)

