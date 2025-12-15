"""Reading comprehension prompt for passage-based questions (FR-08)"""

from src.core.models import Question


class ReadingPromptBuilder:
    """Builds prompt for reading comprehension questions"""
    
    @staticmethod
    def build(question: Question) -> str:
        """
        Build reading comprehension prompt
        
        Strategy: Treat question text as passage, answer based only on provided text
        
        Args:
            question: Question object (question field contains passage)
            
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = f"""Bạn là một chuyên gia phân tích văn bản. Hãy đọc đoạn văn bản được cung cấp và trả lời câu hỏi dựa HOÀN TOÀN trên thông tin trong đoạn văn.

ĐỀ BÀI:
{question.question}

CÁC LỰA CHỌN:
{choices_text}

HƯỚNG DẪN:
1. Đọc kỹ toàn bộ đoạn văn bản
2. Xác định thông tin cần thiết để trả lời câu hỏi
3. Chỉ sử dụng thông tin từ đoạn văn, không sử dụng kiến thức bên ngoài
4. Chọn đáp án chính xác nhất (A, B, C hoặc D)
5. Giải thích ngắn gọn tại sao bạn chọn đáp án đó dựa trên đoạn văn"""
        
        return prompt
    
    @staticmethod
    def extract_passage(question: Question) -> str:
        """
        Extract passage from question text
        
        Assumes passage is provided in question field with markers like:
        [1], [Context:], etc.
        
        Args:
            question: Question object
            
        Returns:
            Extracted passage text
        """
        # The passage is already in the question field for reading mode
        # Just return it as-is
        return question.question

