"""Chain-of-Thought prompt for STEM questions (FR-08)"""

from src.core.models import Question


class STEMPromptBuilder:
    """Builds CoT prompt for math/science questions"""
    
    @staticmethod
    def build(question: Question) -> str:
        """
        Build Chain-of-Thought prompt for STEM question
        
        Instruction: "Let's think step by step..."
        
        Args:
            question: Question object
            
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = f"""Bạn là một chuyên gia toán học và khoa học. Hãy giải quyết câu hỏi này bằng cách suy nghĩ từng bước.

Câu hỏi: {question.question}

Các lựa chọn:
{choices_text}

Hãy suy nghĩ từng bước:
Bước 1: Xác định thông tin được cung cấp
Bước 2: Xác định những gì chúng ta cần tìm
Bước 3: Lựa chọn công thức hoặc phương pháp phù hợp
Bước 4: Thực hiện các tính toán
Bước 5: Kiểm tra kết quả

Cuối cùng, chọn đáp án đúng nhất (A, B, C hoặc D) và giải thích ngắn gọn."""
        
        return prompt
    
    @staticmethod
    def build_with_context(question: Question, context: str) -> str:
        """
        Build CoT prompt with additional context
        
        Args:
            question: Question object
            context: Additional context/formula reference
            
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = f"""Bạn là một chuyên gia toán học và khoa học. Hãy giải quyết câu hỏi này bằng cách suy nghĩ từng bước.

Thông tin tham khảo:
{context}

Câu hỏi: {question.question}

Các lựa chọn:
{choices_text}

Hãy suy nghĩ từng bước và chọn đáp án đúng nhất (A, B, C hoặc D)."""
        
        return prompt

