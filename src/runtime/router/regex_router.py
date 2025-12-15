"""Regex-based question router (FR-06)"""

import re
from typing import List, Optional
from src.core.constants import READING_PATTERNS, STEM_PATTERNS, RouteMode
from src.core.models import QueryRoute, Question


class RegexRouter:
    """Routes questions to READING, STEM, or RAG mode based on patterns"""
    
    def __init__(
        self,
        reading_patterns: Optional[List[str]] = None,
        stem_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize router with regex patterns
        
        Args:
            reading_patterns: Patterns for detecting reading comprehension
            stem_patterns: Patterns for detecting STEM questions
        """
        self.reading_patterns = reading_patterns or READING_PATTERNS
        self.stem_patterns = stem_patterns or STEM_PATTERNS
        
        # Compile regex patterns for efficiency
        self.reading_compiled = [re.compile(p, re.IGNORECASE) for p in self.reading_patterns]
        self.stem_compiled = [re.compile(p, re.IGNORECASE) for p in self.stem_patterns]
    
    def route(self, question: Question) -> str:
        """
        Route question to appropriate mode
        
        Args:
            question: Question to route
            
        Returns:
            Route mode: "READING", "STEM", or "RAG"
        """
        combined_text = f"{question.question} {' '.join(question.choices)}".lower()
        
        # Check READING patterns first (highest priority)
        if self._matches_patterns(combined_text, self.reading_compiled):
            return RouteMode.READING
        
        # Check STEM patterns
        if self._matches_patterns(combined_text, self.stem_compiled):
            return RouteMode.STEM
        
        # Default to RAG
        return RouteMode.RAG
    
    def route_with_context(self, question: Question) -> QueryRoute:
        """
        Route question and prepare context structure
        
        Args:
            question: Question to route
            
        Returns:
            QueryRoute with mode and empty context (to be filled by RAG)
        """
        mode = self.route(question)
        return QueryRoute(mode=mode)
    
    @staticmethod
    def _matches_patterns(text: str, compiled_patterns: List) -> bool:
        """
        Check if text matches any of the patterns
        
        Args:
            text: Text to check
            compiled_patterns: List of compiled regex patterns
            
        Returns:
            True if any pattern matches
        """
        return any(pattern.search(text) for pattern in compiled_patterns)

