from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ProviderResult:
    """Result from a single prompt provider."""
    provider_id: str
    content: str
    generation_time_ms: int
    success: bool
    error: Optional[Exception] = None


@dataclass
class PromptGenerationResult:
    """Result of system prompt generation."""
    content: str
    provider_results: List[ProviderResult] = field(default_factory=list)
    generation_time_ms: int = 0
    success: bool = True
    errors: List[Exception] = field(default_factory=list)
