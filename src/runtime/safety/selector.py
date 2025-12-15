"""Safety answer selector for handling banned questions (FR-05)"""

import re
from typing import Optional, List
from src.core.constants import BAD_KEYWORDS
from src.core.models import Question
from src.runtime.llm.base import LLMService, LLMServiceResponse


class SafetySelector:
    """Selects appropriate answer when question is flagged as unsafe"""
    
    def __init__(
        self,
        llm_service: LLMService,
        bad_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize safety selector
        
        Args:
            llm_service: LLM service for fallback selection
            bad_keywords: Keywords indicating refusal/prohibition
        """
        self.llm_service = llm_service
        self.bad_keywords = bad_keywords or BAD_KEYWORDS
        
        # Compile regex patterns for efficiency
        self.keyword_patterns = [re.compile(keyword, re.IGNORECASE) for keyword in self.bad_keywords]
    
    async def select_answer(self, question: Question) -> str:
        """
        Select answer for unsafe question
        
        Strategy:
        1. Scan options A-D for refusal keywords
        2. If found, return that option
        3. Otherwise, use LLM fallback
        
        Args:
            question: Question with choices
            
        Returns:
            Selected answer letter (A/B/C/D)
        """
        # Step 1: Regex scan for refusal keywords
        answer = self._scan_for_refusal(question)
        if answer:
            return answer
        
        # Step 2: Fallback to LLM
        answer = await self._llm_fallback(question)
        return answer
    
    def _scan_for_refusal(self, question: Question) -> Optional[str]:
        """
        Scan question options for refusal keywords
        
        Args:
            question: Question to scan
            
        Returns:
            Answer letter (A/B/C/D) if refusal found, None otherwise
        """
        answers = ["A", "B", "C", "D"]
        
        for i, choice in enumerate(question.choices):
            if i >= len(answers):
                break
            
            # Check if choice contains any bad keyword
            if self._contains_bad_keyword(choice):
                return answers[i]
        
        return None
    
    def _contains_bad_keyword(self, text: str) -> bool:
        """
        Check if text contains any bad keyword
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains bad keyword
        """
        return any(pattern.search(text) for pattern in self.keyword_patterns)
    
    async def _llm_fallback(self, question: Question) -> str:
        """
        Use LLM to select refusal answer
        
        Args:
            question: Question to answer
            
        Returns:
            Selected answer letter (A/B/C/D)
        """
        system_prompt = """Bạn là một trợ lý an toàn. Câu hỏi sau đây đã được xác định là nguy hiểm.
Hãy chọn lựa chọn phản ánh việc từ chối trợ giúp cho các hành động bất hợp pháp.
Chỉ trả lời bằng một chữ: A, B, C, hoặc D"""
        
        choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(question.choices)])
        user_input = f"Câu hỏi: {question.question}\n\nLựa chọn:\n{choices_text}"
        
        try:
            response = await self.llm_service.generate(
                user_input=user_input,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=10,
            )
            
            # Extract answer letter from response
            answer = self._extract_answer(response.content)
            return answer if answer else "A"
        except Exception:
            # Default to A on error
            return "A"
    
    @staticmethod
    def _extract_answer(response: str) -> Optional[str]:
        """
        Extract answer letter from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Answer letter or None
        """
        for char in response.upper():
            if char in "ABCD":
                return char
        return None

