from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PromptGenerationResult:
    content: str
    provider_results: List[ProviderResult] = field(default_factory=list)  # pyright: ignore[reportUndefinedVariable]
    generation_time_ms: int
    success: bool
    errors: List[Exception] = field(default_factory=list)

@dataclass
class ProviderResult:
    provider_id: str
    content: str
    generation_time_ms: int
    success: bool
    error: Optional[Exception] = None