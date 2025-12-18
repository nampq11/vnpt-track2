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
    def _clean_choices(choices: List[str], qid: str = "") -> tuple[List[str], str]:
        """
        Clean malformed choices arrays that contain question fragments.
        
        Returns:
            tuple: (cleaned_choices, question_continuation)
            - cleaned_choices: List of actual answer choices
            - question_continuation: Text to append to question (LaTeX, question fragments)
        
        Heuristic: Real choices are typically:
        - Non-empty strings
        - Don't start with LaTeX symbols (=, &, \\, $$)
        - Don't end with question marks (those are question continuations)
        """
        if len(choices) <= 4:
            # Standard format, no cleaning needed
            return choices, ""
        
        question_parts = []  # Fragments to add to question
        cleaned = []  # Actual answer choices
        
        for i, choice in enumerate(choices):
            # Skip empty strings but track them
            if not choice or not choice.strip():
                continue
            
            stripped = choice.strip()
            
            # Check if it's a LaTeX fragment
            is_latex = stripped.startswith(('=', '&', '\\', '$$', 'begin{', 'end{'))
            
            # Check if it's a question fragment (ends with ?)
            is_question = stripped.endswith('?')
            
            # If it looks like question content, add to question_parts
            if is_latex or is_question:
                question_parts.append(choice)
            else:
                # Otherwise it's a real choice
                cleaned.append(choice)
        
        # If we filtered down to a reasonable number (4-10 choices), use cleaned
        if 4 <= len(cleaned) <= 26:
            question_continuation = "\n".join(question_parts) if question_parts else ""
            
            if len(cleaned) != len(choices):
                from loguru import logger
                logger.warning(
                    f"Cleaned malformed choices for {qid}: "
                    f"{len(choices)} → {len(cleaned)} choices"
                )
                if question_continuation:
                    logger.debug(f"Appending {len(question_parts)} fragments to question")
            
            return cleaned, question_continuation
        
        # Otherwise, return original (better to have malformed than empty)
        return choices, ""
    
    @staticmethod
    def load_questions(file_path: str) -> List[Question]:
        """Load questions from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            raw_choices = item.get('choices', [])
            qid = item.get('qid', '')
            original_question = item.get('question', '')
            
            # Clean malformed choices and get question continuation
            cleaned_choices, question_continuation = QuestionProcessor._clean_choices(raw_choices, qid)
            
            # Append question continuation if exists
            if question_continuation:
                complete_question = original_question + "\n" + question_continuation
            else:
                complete_question = original_question
            
            q = Question(
                qid=qid,
                question=complete_question,
                choices=cleaned_choices,
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

