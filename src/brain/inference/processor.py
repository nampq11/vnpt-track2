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
        prompt = f"""Câu hỏi: {question.question}

Các lựa chọn:
A) {question.choices[0] if len(question.choices) > 0 else ''}
B) {question.choices[1] if len(question.choices) > 1 else ''}
C) {question.choices[2] if len(question.choices) > 2 else ''}
D) {question.choices[3] if len(question.choices) > 3 else ''}

Hãy chọn một đáp án đúng nhất (A, B, C hoặc D) và giải thích ngắn gọn lý do của bạn."""
        
        return prompt
    
    @staticmethod
    def parse_answer(response: str) -> str:
        """Extract answer from LLM response"""
        response_upper = response.upper()
        
        # Look for patterns like "Đáp án: A", "Answer: A", etc.
        for answer_char in ['A', 'B', 'C', 'D']:
            if f"ĐÁP ÁN: {answer_char}" in response_upper:
                return answer_char
            if f"ANSWER: {answer_char}" in response_upper:
                return answer_char
            if f"LỰA CHỌN: {answer_char}" in response_upper:
                return answer_char
        
        # Look for standalone answer characters at the start
        words = response_upper.split()
        if words and words[0] in ['A', 'B', 'C', 'D']:
            return words[0]
        
        # Look for any A, B, C, D in the response
        for char in response_upper:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        # Default to first choice if can't parse
        return 'A'

