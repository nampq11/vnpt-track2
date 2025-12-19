"""Vietnamese text preprocessing using underthesea.

Provides text normalization, cleaning, and tokenization for:
- Document processing before embedding
- Query processing before retrieval
- FTS-optimized tokenization
"""

import re
from typing import Optional
from loguru import logger

# Lazy import underthesea to avoid slow startup
_text_normalize = None
_word_tokenize = None


def _get_text_normalize():
    """Lazy load text_normalize from underthesea."""
    global _text_normalize
    if _text_normalize is None:
        from underthesea import text_normalize
        _text_normalize = text_normalize
    return _text_normalize


def _get_word_tokenize():
    """Lazy load word_tokenize from underthesea."""
    global _word_tokenize
    if _word_tokenize is None:
        from underthesea import word_tokenize
        _word_tokenize = word_tokenize
    return _word_tokenize


class VietnameseTextPreprocessor:
    """Preprocess Vietnamese text for embedding and search."""
    
    def __init__(
        self,
        normalize: bool = True,
        remove_urls: bool = True,
        remove_special: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalize: Whether to normalize Vietnamese diacritics
            remove_urls: Whether to remove URLs from text
            remove_special: Whether to remove special characters
        """
        self.normalize = normalize
        self.remove_urls = remove_urls
        self.remove_special = remove_special
    
    def clean_document(self, text: str) -> str:
        """
        Clean document text before chunking/embedding.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # 1. Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'URL:.*?\n', '', text)
        
        # 2. Normalize Vietnamese diacritics
        if self.normalize:
            try:
                text_normalize = _get_text_normalize()
                text = text_normalize(text)
            except Exception as e:
                logger.debug(f"text_normalize failed: {e}")
        
        # 3. Remove special chars (keep Vietnamese letters and punctuation)
        if self.remove_special:
            # Keep: word chars, spaces, Vietnamese diacritics, common punctuation
            text = re.sub(r'[^\w\sÀ-ỹđĐ.,!?;\-–—:()\'\"]+', ' ', text)
        
        # 4. Collapse multiple whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_query(self, query: str) -> str:
        """
        Clean query text before embedding/search.
        
        Args:
            query: Raw query text
            
        Returns:
            Cleaned query
        """
        if not query:
            return ""
        
        # Normalize diacritics
        if self.normalize:
            try:
                text_normalize = _get_text_normalize()
                query = text_normalize(query)
            except Exception as e:
                logger.debug(f"text_normalize failed: {e}")
        
        # Remove special characters (keep Vietnamese and basic punctuation)
        query = re.sub(r'[^\w\sÀ-ỹđĐ.,!?]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query
    
    def tokenize_for_fts(self, text: str) -> str:
        """
        Tokenize text for full-text search.
        
        Joins compound words with underscores for better FTS matching.
        Example: "Hồ Chí Minh" → "Hồ_Chí_Minh"
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tokenized text with underscores
        """
        if not text:
            return ""
        
        try:
            word_tokenize = _get_word_tokenize()
            # format="text" joins compound words with underscores
            tokens = word_tokenize(text, format="text")
            return tokens
        except Exception as e:
            logger.debug(f"word_tokenize failed: {e}")
            return text


# Singleton instance for reuse
_preprocessor: Optional[VietnameseTextPreprocessor] = None


def get_preprocessor() -> VietnameseTextPreprocessor:
    """Get singleton preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = VietnameseTextPreprocessor()
    return _preprocessor


def clean_document(text: str) -> str:
    """Clean document text before chunking/embedding."""
    return get_preprocessor().clean_document(text)


def clean_query(query: str) -> str:
    """Clean query text before embedding/search."""
    return get_preprocessor().clean_query(query)


def tokenize_for_fts(text: str) -> str:
    """Tokenize text for full-text search."""
    return get_preprocessor().tokenize_for_fts(text)

