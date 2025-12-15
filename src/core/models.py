"""Pydantic models for Titan Shield RAG System"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class ChunkType(str, Enum):
    """Document chunk type classification"""
    LAW = "LAW"
    MATH = "MATH"
    HISTORY = "HISTORY"
    GENERAL = "GENERAL"


@dataclass
class ChunkMetadata:
    """Metadata for document chunks - matches PRD 4.1 specification"""
    source: str
    type: ChunkType
    valid_from: int = 1900
    expire_at: int = 9999
    province: str = "ALL"


@dataclass
class Chunk:
    """Document chunk with content and metadata"""
    id: str
    text: str
    metadata: ChunkMetadata


@dataclass
class Question:
    """Multiple-choice question structure"""
    qid: str
    question: str
    choices: List[str]
    answer: Optional[str] = None


@dataclass
class PredictionResult:
    """Model prediction result"""
    qid: str
    predicted_answer: str
    confidence: float = 0.0
    reasoning: Optional[str] = None
    route_mode: Optional[str] = None  # READING, STEM, or RAG


@dataclass
class QueryRoute:
    """Result of routing query to appropriate handler"""
    mode: str  # "READING", "STEM", or "RAG"
    context_chunks: List[Chunk] = field(default_factory=list)
    extracted_year: Optional[int] = None


@dataclass
class SafetyCheckResult:
    """Result of safety guardrail check"""
    is_safe: bool
    similarity_score: float
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Single search result from hybrid retrieval"""
    chunk: Chunk
    keyword_score: float
    semantic_score: float
    final_score: float
    rank: int


@dataclass
class HybridSearchResults:
    """Results from hybrid BM25 + Vector search"""
    results: List[SearchResult] = field(default_factory=list)
    total_retrieved: int = 0
    temporal_filter_applied: bool = False

