"""RAG prompt with negative constraints (FR-08)"""

from typing import List
from src.core.models import Question, SearchResult


class RAGPromptBuilder:
    """Builds RAG prompt with context and negative constraints"""
    
    @staticmethod
    def build(question: Question, search_results: List[SearchResult]) -> str:
        """
        Build RAG prompt with retrieved context and negative constraints
        
        Constraint: "Only answer based on context. If info is missing, state 'No information'."
        
        Args:
            question: Question object
            search_results: List of retrieved SearchResult objects
            
        Returns:
            Formatted prompt string
        """
        # Build context from search results
        context_parts = []
        for result in search_results:
            context_parts.append(f"[{result.chunk.id}] {result.chunk.text}")
        
        context_text = "\n\n".join(context_parts) if context_parts else "Không có thông tin liên quan"
        
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = f"""Bạn là một trợ lý thông minh chuyên trả lời câu hỏi trắc nghiệm dựa trên thông tin được cung cấp.

LƯU Ý QUAN TRỌNG:
- Chỉ trả lời dựa trên thông tin trong ngữ cảnh được cung cấp dưới đây.
- Nếu thông tin không có trong ngữ cảnh, hãy nêu rõ "Thông tin không có sẵn".
- Đừng sử dụng kiến thức bên ngoài.
- Hãy chọn đáp án phù hợp nhất dựa trên ngữ cảnh được cung cấp.

NGỮ CẢNH:
{context_text}

CÂUHỎI: {question.question}

CÁC LỰA CHỌN:
{choices_text}

HƯỚNG DẪN:
1. Đọc kỹ ngữ cảnh
2. Phân tích câu hỏi
3. Tìm thông tin liên quan trong ngữ cảnh
4. Chọn đáp án chính xác nhất (A, B, C hoặc D)
5. Giải thích ngắn gọn dựa trên ngữ cảnh"""
        
        return prompt
    
    @staticmethod
    def build_simple(question: Question, context: str) -> str:
        """
        Build simple RAG prompt with raw context text
        
        Args:
            question: Question object
            context: Context text
            
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = f"""Dựa trên thông tin được cung cấp, hãy trả lời câu hỏi sau:

NGỮ CẢNH:
{context}

CÂUHỎI: {question.question}

CÁC LỰA CHỌN:
{choices_text}

Chỉ sử dụng thông tin từ ngữ cảnh để trả lời. Chọn đáp án A, B, C hoặc D."""
        
        return prompt

