"""Vietnamese text cleaning and normalization"""

import re
import unicodedata
from typing import List


class TextCleaner:
    """Cleans and normalizes Vietnamese text"""
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Vietnamese Unicode (NFD -> NFC)
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Decompose combining characters (NFD)
        decomposed = unicodedata.normalize('NFD', text)
        # Recompose (NFC) for canonical form
        normalized = unicodedata.normalize('NFC', decomposed)
        return normalized
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        Remove extra whitespace and normalize line breaks
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        Remove HTML tags from text
        
        Args:
            text: Input text with HTML
            
        Returns:
            Text without HTML tags
        """
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&quot;', '"')
        return text
    
    @staticmethod
    def remove_special_chars(text: str, keep_punctuation: bool = True) -> str:
        """
        Remove special characters
        
        Args:
            text: Input text
            keep_punctuation: Whether to keep punctuation marks
            
        Returns:
            Cleaned text
        """
        if keep_punctuation:
            # Keep Vietnamese characters, spaces, and basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\-—–"\'\(\)\[\]]+', '', text, flags=re.UNICODE)
        else:
            # Remove all non-alphanumeric
            text = re.sub(r'[^\w\s]+', '', text, flags=re.UNICODE)
        return text
    
    @staticmethod
    def clean(text: str, remove_html: bool = True) -> str:
        """
        Complete text cleaning pipeline
        
        Args:
            text: Input text
            remove_html: Whether to remove HTML tags
            
        Returns:
            Cleaned text
        """
        # Normalize unicode
        text = TextCleaner.normalize_unicode(text)
        
        # Remove HTML if needed
        if remove_html:
            text = TextCleaner.remove_html_tags(text)
        
        # Remove extra whitespace
        text = TextCleaner.remove_extra_whitespace(text)
        
        return text
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 128,
    ) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Input text
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find chunk end
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence ending within last 50 chars
                last_period = text.rfind('.', end - 50, end)
                last_newline = text.rfind('\n', end - 50, end)
                last_break = max(last_period, last_newline)
                
                if last_break > start:
                    end = last_break + 1
            
            chunks.append(text[start:end].strip())
            
            # Move start position (with overlap)
            start = end - overlap
        
        return [c for c in chunks if c]  # Filter empty chunks

