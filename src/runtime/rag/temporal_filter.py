"""Temporal filtering for time-sensitive documents (FR-07)"""

import re
from typing import List, Optional
from src.core.constants import YEAR_EXTRACTION_PATTERNS
from src.core.models import Chunk


class TemporalFilter:
    """Filters document chunks based on validity dates"""
    
    def __init__(self, year_patterns: Optional[List[str]] = None):
        """
        Initialize temporal filter
        
        Args:
            year_patterns: Regex patterns for year extraction
        """
        self.year_patterns = year_patterns or YEAR_EXTRACTION_PATTERNS
        self.compiled_patterns = [re.compile(p) for p in self.year_patterns]
    
    def extract_year(self, text: str) -> Optional[int]:
        """
        Extract year from text
        
        Args:
            text: Text to extract year from
            
        Returns:
            Year as integer or None
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                try:
                    year_str = match.group(1) if match.lastindex else match.group(0)
                    year = int(year_str)
                    # Validate year is reasonable (1900-2100)
                    if 1900 <= year <= 2100:
                        return year
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def filter_chunks(
        self,
        chunks: List[Chunk],
        query_year: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Filter chunks based on temporal validity
        
        Args:
            chunks: List of chunks to filter
            query_year: Reference year from query (extracted via extract_year)
            
        Returns:
            Filtered list of valid chunks
        """
        if query_year is None:
            # No temporal constraint, return all
            return chunks
        
        filtered = []
        for chunk in chunks:
            # Check if chunk is valid for the query year
            if chunk.metadata.valid_from <= query_year <= chunk.metadata.expire_at:
                filtered.append(chunk)
        
        return filtered
    
    def filter_and_rank(
        self,
        chunks: List[Chunk],
        query_year: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Filter chunks and optionally rank by recency
        
        Args:
            chunks: List of chunks to filter
            query_year: Reference year
            
        Returns:
            Filtered and ranked chunks (most recent first)
        """
        filtered = self.filter_chunks(chunks, query_year)
        
        # Sort by valid_from descending (most recent first)
        filtered.sort(key=lambda c: c.metadata.valid_from, reverse=True)
        
        return filtered

