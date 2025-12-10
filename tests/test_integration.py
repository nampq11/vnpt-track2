"""Integration tests for inference pipeline"""
import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.brain.inference.processor import Question, QuestionProcessor
from src.brain.inference.evaluator import Evaluator


class TestIntegration:
    """Integration tests"""
    
    @pytest.fixture
    def temp_data(self):
        """Create temporary test data"""
        test_data = [
            {
                "qid": "q1",
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": "C"
            },
            {
                "qid": "q2",
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
                "answer": "B"
            },
        ]
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            encoding='utf-8'
        ) as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        Path(temp_file).unlink()
    
    def test_load_questions(self, temp_data):
        """Test loading questions from file"""
        processor = QuestionProcessor()
        questions = processor.load_questions(temp_data)
        
        assert len(questions) == 2
        assert questions[0].qid == "q1"
        assert questions[1].qid == "q2"
        assert len(questions[0].choices) == 4
    
    def test_format_and_parse_flow(self):
        """Test formatting and parsing flow"""
        q = Question(
            qid="test",
            question="Test question?",
            choices=["A", "B", "C", "D"]
        )
        
        processor = QuestionProcessor()
        
        # Format
        prompt = processor.format_for_llm(q)
        assert "Test question?" in prompt
        
        # Simulate LLM response
        response = "The answer is definitely B"
        answer = processor.parse_answer(response)
        assert answer == "B"
    
    def test_evaluation_flow(self):
        """Test full evaluation flow"""
        predictions = [
            {"qid": "q1", "predicted_answer": "C"},
            {"qid": "q2", "predicted_answer": "B"},
        ]
        ground_truth = ["C", "B"]
        
        metrics = Evaluator.evaluate(predictions, ground_truth)
        
        assert metrics.accuracy == 1.0
        assert len(metrics.details) == 2
        assert all(d["correct"] for d in metrics.details)
    
    def test_end_to_end_with_file(self, temp_data):
        """Test end-to-end with file I/O"""
        processor = QuestionProcessor()
        
        # Load
        questions = processor.load_questions(temp_data)
        assert len(questions) == 2
        
        # Process
        predictions = []
        for q in questions:
            prompt = processor.format_for_llm(q)
            # Simulate answer parsing
            answer = q.answer if q.answer else "A"
            predictions.append({
                "qid": q.qid,
                "predicted_answer": answer
            })
        
        # Evaluate
        ground_truth = [q.answer for q in questions]
        metrics = Evaluator.evaluate(predictions, ground_truth)
        
        assert metrics.accuracy == 1.0
        assert metrics.correct_answers == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

