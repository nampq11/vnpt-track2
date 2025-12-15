"""Base crawler class with rate limiting"""

from abc import ABC, abstractmethod
from typing import List
import asyncio
from datetime import datetime


class BaseCrawler(ABC):
    """Abstract base class for data crawlers"""
    
    def __init__(
        self,
        max_concurrent: int = 5,
        delay_between_requests: float = 0.5,
    ):
        """
        Initialize crawler with rate limiting
        
        Args:
            max_concurrent: Maximum concurrent requests
            delay_between_requests: Delay between requests in seconds
        """
        self.max_concurrent = max_concurrent
        self.delay_between_requests = delay_between_requests
        self.session = None
    
    @abstractmethod
    async def crawl(self, url: str) -> str:
        """
        Crawl single URL and return content
        
        Args:
            url: URL to crawl
            
        Returns:
            HTML or text content
        """
        pass
    
    @abstractmethod
    async def parse(self, content: str) -> List[str]:
        """
        Parse content into text chunks
        
        Args:
            content: Raw content from crawl
            
        Returns:
            List of text chunks
        """
        pass
    
    async def crawl_multiple(self, urls: List[str]) -> List[str]:
        """
        Crawl multiple URLs with concurrency control
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of parsed chunks from all URLs
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def crawl_with_limit(url: str) -> str:
            async with semaphore:
                await asyncio.sleep(self.delay_between_requests)
                try:
                    return await self.crawl(url)
                except Exception as e:
                    print(f"Failed to crawl {url}: {str(e)}")
                    return ""
        
        contents = await asyncio.gather(*[crawl_with_limit(url) for url in urls])
        
        # Parse all contents
        all_chunks = []
        for content in contents:
            if content:
                chunks = await self.parse(content)
                all_chunks.extend(chunks)
        
        return all_chunks
    
    async def close(self):
        """Close crawler session"""
        if self.session:
            await self.session.close()

