"""Question preprocessing and formatting for inference"""
from dataclasses import dataclass
from typing import List
import json


@dataclass
class Question:
    qid: str
    question: str
    choices: List[str]
    answer: str = ""  # Only for validation


@dataclass
class PredictionResult:
    qid: str
    predicted_answer: str
    confidence: float = 0.0


class QuestionProcessor:
    """Process and format questions for LLM inference"""
    
    @staticmethod
    def load_questions(file_path: str) -> List[Question]:
        """Load questions from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            q = Question(
                qid=item.get('qid', ''),
                question=item.get('question', ''),
                choices=item.get('choices', []),
                answer=item.get('answer', '')
            )
            questions.append(q)
        
        return questions
    
    @staticmethod
    def format_for_llm(question: Question) -> str:
        """Format question and choices for LLM input"""
        # Dynamically format all choices
        choices_text = "\n".join([
            f"{chr(65 + i)}) {choice}" 
            for i, choice in enumerate(question.choices)
        ])
        
        # Generate choice letters list for prompt
        num_choices = len(question.choices)
        if num_choices <= 4:
            choice_range = "A, B, C hoặc D"
        else:
            last_letter = chr(64 + num_choices)  # 65=A, so 64+n gives nth letter
            choice_range = f"A đến {last_letter}"
        
        prompt = f"""Câu hỏi: {question.question}

Các lựa chọn:
{choices_text}

Hãy chọn một đáp án đúng nhất ({choice_range}) và giải thích ngắn gọn lý do của bạn."""
        
        return prompt
    
    @staticmethod
    def parse_answer(response: str) -> str:
        """Extract answer from LLM response"""
        response_upper = response.upper()
        
        # All possible answer letters A-Z
        import re
        
        # Look for explicit patterns with answer markers
        # Pattern 1: "Đáp án: A" or "Đáp án đúng nhất: A" or similar
        patterns = [
            r'ĐÁP ÁN[^:]*:\s*\*?\*?([A-Z])\)',
            r'ĐÁP ÁN[^:]*:\s*\*?\*?([A-Z])\b',
            r'ANSWER[^:]*:\s*\*?\*?([A-Z])\)',
            r'ANSWER[^:]*:\s*\*?\*?([A-Z])\b',
            r'LỰA CHỌN[^:]*:\s*\*?\*?([A-Z])\)',
            r'LỰA CHỌN[^:]*:\s*\*?\*?([A-Z])\b',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                return match.group(1)
        
        # Look for "**A)**" or "*A)*" patterns (markdown bold)
        match = re.search(r'\*+([A-Z])\)\*+', response_upper)
        if match:
            return match.group(1)
        
        # Look for standalone answer at start of response
        match = re.match(r'^([A-Z])\)', response_upper.strip())
        if match:
            return match.group(1)
        
        # Look for first letter followed by closing paren in first 200 chars
        # This catches "A) explanation" patterns
        first_part = response_upper[:200]
        match = re.search(r'\b([A-Z])\)', first_part)
        if match:
            return match.group(1)
        
        # Default to 'A' if can't parse
        return 'A'

