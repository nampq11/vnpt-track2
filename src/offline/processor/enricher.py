"""Metadata enrichment using LLM (GPT-4 / Azure)"""

from typing import List, Optional
import asyncio
from src.core.models import Chunk, ChunkMetadata, ChunkType
from src.runtime.llm.base import LLMService


class Enricher:
    """Enriches document chunks with metadata using LLM"""
    
    def __init__(
        self,
        llm_service: LLMService,
        max_concurrent: int = 5,
    ):
        """
        Initialize enricher
        
        Args:
            llm_service: LLM service (Azure/GPT-4 for offline)
            max_concurrent: Maximum concurrent enrichment requests
        """
        self.llm_service = llm_service
        self.max_concurrent = max_concurrent
    
    async def enrich_chunk(
        self,
        chunk_id: str,
        text: str,
    ) -> ChunkMetadata:
        """
        Enrich single chunk with metadata
        
        Uses LLM to tag:
        - type: LAW, MATH, HISTORY, GENERAL
        - valid_from: Year when info becomes valid
        
        Args:
            chunk_id: Unique chunk identifier
            text: Chunk text content
            
        Returns:
            ChunkMetadata with enriched fields
        """
        prompt = f"""Phân tích đoạn văn bản sau và trích xuất metadata:

TEXT:
{text[:1000]}  # Limit to 1000 chars for speed

Hãy xác định:
1. Type (LAW/MATH/HISTORY/GENERAL)
2. Valid from year (khi thông tin này có hiệu lực)
3. Source/Topic

Trả lời trong format:
TYPE: [LAW|MATH|HISTORY|GENERAL]
VALID_FROM: [year or 1900]
SOURCE: [brief source]"""
        
        try:
            response = await self.llm_service.generate(
                user_input=prompt,
                temperature=0.3,
                max_tokens=50,
            )
            
            # Parse response
            content = response.content
            doc_type = self._extract_type(content)
            valid_from = self._extract_year(content)
            
            return ChunkMetadata(
                source=chunk_id.split('_')[0] if '_' in chunk_id else "UNKNOWN",
                type=doc_type,
                valid_from=valid_from,
                expire_at=9999,  # Default: never expires
                province="ALL",
            )
        except Exception as e:
            print(f"Enrichment error for {chunk_id}: {str(e)}")
            # Return default metadata on error
            return ChunkMetadata(
                source="UNKNOWN",
                type=ChunkType.GENERAL,
                valid_from=1900,
            )
    
    async def enrich_batch(
        self,
        chunks: List[tuple],  # List of (chunk_id, text) tuples
    ) -> List[ChunkMetadata]:
        """
        Enrich multiple chunks with concurrency control
        
        Args:
            chunks: List of (chunk_id, text) tuples
            
        Returns:
            List of enriched ChunkMetadata
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def enrich_with_limit(chunk_id: str, text: str) -> ChunkMetadata:
            async with semaphore:
                await asyncio.sleep(0.1)  # Rate limiting
                return await self.enrich_chunk(chunk_id, text)
        
        tasks = [enrich_with_limit(cid, text) for cid, text in chunks]
        metadata_list = await asyncio.gather(*tasks)
        
        return metadata_list
    
    @staticmethod
    def _extract_type(response: str) -> ChunkType:
        """Extract type from LLM response"""
        response_upper = response.upper()
        if "LAW" in response_upper:
            return ChunkType.LAW
        elif "MATH" in response_upper:
            return ChunkType.MATH
        elif "HISTORY" in response_upper:
            return ChunkType.HISTORY
        else:
            return ChunkType.GENERAL
    
    @staticmethod
    def _extract_year(response: str) -> int:
        """Extract valid_from year from LLM response"""
        import re
        match = re.search(r'\d{4}', response)
        if match:
            year = int(match.group())
            if 1900 <= year <= 2100:
                return year
        return 1900

