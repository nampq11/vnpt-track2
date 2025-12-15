"""Constants for Titan Shield RAG System - matches PRD specifications"""

from enum import Enum
from typing import List

# ============================================================================
# SAFETY CONFIGURATION (FR-04, FR-05)
# ============================================================================

# Cosine similarity threshold for safety check (PRD 4.1)
SAFETY_THRESHOLD: float = 0.85

# Keywords that indicate a safety refusal option (Vietnamese)
BAD_KEYWORDS: List[str] = [
    "không được phép",
    "bị nghiêm cấm",
    "vi phạm",
    "từ chối",
    "cấm",
    "illegal",
    "unlawful",
    "prohibited",
]

# ============================================================================
# ROUTER CONFIGURATION (FR-06)
# ============================================================================

# Regex patterns for READING mode detection
READING_PATTERNS: List[str] = [
    r"đoạn văn",
    r"bài đọc",
    r"đoạn thông tin",
    r"context:",
    r"\[1\]",
    r"passage:",
    r"text:",
]

# Regex patterns for STEM mode detection (math, science)
STEM_PATTERNS: List[str] = [
    r"\\int",
    r"\\sum",
    r"\\frac",
    r"tính",
    r"giá trị",
    r"hàm số",
    r"phương trình",
    r"tích phân",
    r"đạo hàm",
    r"lim",
    r"\^",
    r"√",
]

# ============================================================================
# RAG CONFIGURATION (FR-07)
# ============================================================================

# Maximum number of chunks to retrieve
MAX_CHUNKS_RETRIEVED: int = 5

# Minimum relevance score to include chunk (0-1)
MIN_RELEVANCE_SCORE: float = 0.3

# Weights for RRF (Reciprocal Rank Fusion)
BM25_WEIGHT: float = 0.6
VECTOR_WEIGHT: float = 0.4

# ============================================================================
# TEMPORAL FILTERING (FR-07)
# ============================================================================

# Regex patterns to extract year from query
YEAR_EXTRACTION_PATTERNS: List[str] = [
    r"năm (\d{4})",  # "năm 2025"
    r"(\d{4})",       # Any 4-digit number
]

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Rate limiting for concurrent API calls
EMBEDDING_MAX_CONCURRENT: int = 8
CHAT_MAX_CONCURRENT: int = 5

# Retry configuration
RETRY_BACKOFF_TIMES: List[float] = [1.0, 2.0, 4.0]  # Exponential backoff: 1s, 2s, 4s
MAX_RETRIES: int = 3

# ============================================================================
# ARTIFACT PATHS
# ============================================================================

ARTIFACTS_DIR: str = "src/artifacts"
FAISS_INDEX_PATH: str = f"{ARTIFACTS_DIR}/faiss.index"
BM25_INDEX_PATH: str = f"{ARTIFACTS_DIR}/bm25.pkl"
SAFETY_INDEX_PATH: str = f"{ARTIFACTS_DIR}/safety.npy"
METADATA_PATH: str = f"{ARTIFACTS_DIR}/metadata.json"


class RouteMode(str, Enum):
    """Question routing modes"""
    READING = "READING"  # Direct context provided
    STEM = "STEM"        # Math/science with CoT
    RAG = "RAG"         # Knowledge retrieval needed

