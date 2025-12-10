"""Simple test runner for quick inference testing"""
import asyncio
from typing import List, Optional

from src.brain.llm.services.ollama import OllamaService
from src.brain.inference.processor import QuestionProcessor, Question


class SimpleInferenceTest:
    """Simple single-question inference for testing"""
    
    @staticmethod
    async def test_single_question(
        question: Question,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434/v1",
    ) -> str:
        """Test inference on a single question"""
        # Initialize service
        llm_service = OllamaService(
            base_url=base_url,
            api_key="ollama",
            model=model
        )
        
        # Format question
        processor = QuestionProcessor()
        prompt = processor.format_for_llm(question)
        
        # Get response
        response = await llm_service.generate(prompt)
        
        # Parse answer
        answer = processor.parse_answer(response)
        
        return answer
    
    @staticmethod
    async def test_first_n_questions(
        file_path: str,
        n: int = 5,
        model: str = "qwen3:1.7b",
    ) -> None:
        """Test first N questions from dataset"""
        processor = QuestionProcessor()
        questions = processor.load_questions(file_path)[:n]
        
        llm_service = OllamaService(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model=model
        )
        
        print(f"Testing first {len(questions)} questions with model: {model}\n")
        
        correct = 0
        for i, q in enumerate(questions):
            print(f"Q{i+1}: {q.qid}")
            print(f"Question: {q.question[:100]}...")
            print(f"Choices: {q.choices}")
            print(f"Ground truth: {q.answer}")
            
            prompt = processor.format_for_llm(q)
            response = await llm_service.generate(prompt)
            answer = processor.parse_answer(response)
            
            is_correct = answer.upper() == q.answer.upper()
            correct += is_correct
            
            print(f"Predicted: {answer} {'✓' if is_correct else '✗'}")
            print(f"Response preview: {response[:200]}...\n")
        
        print(f"Accuracy: {correct}/{len(questions)} ({100*correct//len(questions)}%)")

