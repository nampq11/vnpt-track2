"""Document processing and chunking for Vietnamese text."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from pathlib import Path
import re
import uuid
import json
from loguru import logger

from src.brain.rag.text_preprocessor import clean_document as preprocess_text


@dataclass
class DocumentChunk:
    """Represents a single chunk of a document."""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def category(self) -> str:
        return self.metadata.get("category", "unknown")
    
    @property
    def title(self) -> str:
        return self.metadata.get("title", "")
    
    @property
    def section(self) -> str:
        return self.metadata.get("section", "")


class DocumentProcessor:
    """Process Vietnamese documents into chunks with metadata."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def process_directory(self, data_dir: Path) -> List[DocumentChunk]:
        """Process all .txt files in data directory."""
        chunks = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        total_files = 0
        for category_dir in data_path.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            
            category = category_dir.name
            
            for file_path in category_dir.glob("*.txt"):
                try:
                    file_chunks = self._process_file(file_path, category)
                    chunks.extend(file_chunks)
                    total_files += 1
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Processed {total_files} files into {len(chunks)} chunks")
        return chunks
    
    def _process_file(
        self,
        file_path: Path,
        category: str,
    ) -> List[DocumentChunk]:
        """Process a single file into chunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Extract metadata
        title = self._extract_title(content)
        sections = self._extract_sections(content)
        
        chunks = []
        for section_name, section_content in sections:
            section_chunks = self._chunk_text(
                text=section_content,
                metadata={
                    "category": category,
                    "title": title,
                    "section": section_name,
                    "source_file": str(file_path),
                }
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _extract_title(self, content: str) -> str:
        """Extract title from document."""
        match = re.search(r"Tiêu đề:\s*(.+?)(?:\n|$)", content)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_sections(
        self,
        content: str,
    ) -> List[Tuple[str, str]]:
        """Extract sections with their content."""
        # Remove header (title, URL, first divider)
        content = re.sub(r"^.*?-{10,}\n", "", content, count=1, flags=re.DOTALL)
        
        # Split by section headers
        section_pattern = r"(?:^|\n)\s*(===\s*.+?\s*===)\s*\n"
        parts = re.split(section_pattern, content)
        
        sections = []
        current_section = "Tóm tắt"  # Default section name
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            if re.match(r"===\s*.+?\s*===", part):
                # This is a section header
                current_section = part.replace("===", "").strip()
                i += 1
            elif part:
                # This is content
                sections.append((current_section, part))
                i += 1
            else:
                i += 1
        
        # If no sections found, treat entire content as one section
        if not sections:
            cleaned = re.sub(r"Tiêu đề:.*?\n", "", content)
            cleaned = re.sub(r"URL:.*?\n", "", cleaned)
            cleaned = re.sub(r"-{10,}", "", cleaned)
            if cleaned.strip():
                sections.append(("content", cleaned.strip()))
        
        return sections
    
    def _chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        # Clean text before chunking (normalize Vietnamese, remove noise)
        text = preprocess_text(text)
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            
            # Try to find sentence boundary
            if end < text_len:
                # Look for sentence endings within last 100 chars
                search_start = max(end - 100, start)
                sentence_end = self._find_sentence_boundary(
                    text[search_start:min(end + 50, text_len)]
                )
                if sentence_end > 0:
                    end = search_start + sentence_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4())[:8],
                    content=chunk_text,
                    metadata=metadata.copy(),
                )
                chunks.append(chunk)
            
            # Move start with overlap
            if end >= text_len:
                break
            start = end - self.overlap
            if start < 0:
                break
        
        return chunks
    
    def _find_sentence_boundary(self, text: str) -> int:
        """Find best sentence boundary position."""
        # Vietnamese sentence endings
        endings = [". ", "! ", "? ", ".\n", "!\n", "?\n", ".", "!", "?"]
        
        best_pos = -1
        for ending in endings:
            pos = text.rfind(ending)
            if pos > best_pos:
                best_pos = pos + len(ending)
        
        return best_pos if best_pos > 0 else -1


def save_chunks(chunks: List[DocumentChunk], output_path: str):
    """Save chunks to JSON file."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "metadata": c.metadata,
        }
        for c in chunks
    ]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks(input_path: str) -> List[DocumentChunk]:
    """Load chunks from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = [
        DocumentChunk(
            chunk_id=d["chunk_id"],
            content=d["content"],
            metadata=d["metadata"],
        )
        for d in data
    ]
    
    logger.info(f"Loaded {len(chunks)} chunks from {input_path}")
    return chunks

