"""Unit tests for inference pipeline"""
import pytest
from src.brain.inference.processor import Question, QuestionProcessor, PredictionResult
from src.brain.inference.evaluator import Evaluator


class TestQuestionProcessor:
    """Test question processing"""
    
    def test_format_for_llm(self):
        """Test question formatting"""
        q = Question(
            qid="test_1",
            question="What is 2+2?",
            choices=["3", "4", "5", "6"]
        )
        
        prompt = QuestionProcessor.format_for_llm(q)
        
        assert "What is 2+2?" in prompt
        assert "A) 3" in prompt
        assert "B) 4" in prompt
        assert "C) 5" in prompt
        assert "D) 6" in prompt
    
    def test_parse_answer_uppercase(self):
        """Test parsing uppercase answers"""
        response = "ĐÁP ÁN: A"
        answer = QuestionProcessor.parse_answer(response)
        assert answer == "A"
    
    def test_parse_answer_mixed(self):
        """Test parsing mixed case answers"""
        response = "Answer: b"
        answer = QuestionProcessor.parse_answer(response)
        assert answer == "B"
    
    def test_parse_answer_standalone(self):
        """Test parsing standalone answer letter"""
        response = "C is the correct answer"
        answer = QuestionProcessor.parse_answer(response)
        assert answer == "C"
    
    def test_parse_answer_default(self):
        """Test default answer when can't parse"""
        response = "không có câu trả lời"
        answer = QuestionProcessor.parse_answer(response)
        assert answer == "A"


class TestEvaluator:
    """Test evaluation metrics"""
    
    def test_perfect_accuracy(self):
        """Test perfect accuracy calculation"""
        predictions = [
            {"qid": "q1", "predicted_answer": "A"},
            {"qid": "q2", "predicted_answer": "B"},
        ]
        ground_truth = ["A", "B"]
        
        metrics = Evaluator.evaluate(predictions, ground_truth)
        
        assert metrics.accuracy == 1.0
        assert metrics.correct_answers == 2
        assert metrics.incorrect_answers == 0
    
    def test_partial_accuracy(self):
        """Test partial accuracy calculation"""
        predictions = [
            {"qid": "q1", "predicted_answer": "A"},
            {"qid": "q2", "predicted_answer": "B"},
            {"qid": "q3", "predicted_answer": "C"},
        ]
        ground_truth = ["A", "C", "C"]
        
        metrics = Evaluator.evaluate(predictions, ground_truth)
        
        assert metrics.accuracy == pytest.approx(2/3)
        assert metrics.correct_answers == 2
        assert metrics.incorrect_answers == 1
    
    def test_case_insensitive(self):
        """Test case insensitive comparison"""
        predictions = [
            {"qid": "q1", "predicted_answer": "a"},
            {"qid": "q2", "predicted_answer": "B"},
        ]
        ground_truth = ["A", "b"]
        
        metrics = Evaluator.evaluate(predictions, ground_truth)
        
        assert metrics.accuracy == 1.0
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched lengths"""
        predictions = [
            {"qid": "q1", "predicted_answer": "A"},
        ]
        ground_truth = ["A", "B"]
        
        with pytest.raises(ValueError):
            Evaluator.evaluate(predictions, ground_truth)


class TestPredictionResult:
    """Test prediction result data class"""
    
    def test_creation(self):
        """Test PredictionResult creation"""
        result = PredictionResult(
            qid="test_1",
            predicted_answer="A",
            confidence=0.95
        )
        
        assert result.qid == "test_1"
        assert result.predicted_answer == "A"
        assert result.confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

