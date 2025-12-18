"""Evaluation metrics and performance tracking"""
from typing import List, Dict
from dataclasses import dataclass
import json


@dataclass
class EvaluationMetrics:
    accuracy: float
    total_questions: int
    correct_answers: int
    incorrect_answers: int
    details: List[Dict] = None


class Evaluator:
    """Evaluate model predictions against ground truth"""
    
    @staticmethod
    def evaluate(
        predictions: List[Dict],
        ground_truth: List[str]
    ) -> EvaluationMetrics:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions: List of dicts with 'qid' and 'predicted_answer'
            ground_truth: List of correct answers
        
        Returns:
            EvaluationMetrics with accuracy and details
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = 0
        details = []
        
        for pred, truth in zip(predictions, ground_truth):
            is_correct = pred['predicted_answer'].upper() == truth.upper()
            correct += is_correct
            
            details.append({
                'qid': pred.get('qid', ''),
                'predicted': pred['predicted_answer'],
                'ground_truth': truth,
                'correct': is_correct
            })
        
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            total_questions=total,
            correct_answers=correct,
            incorrect_answers=total - correct,
            details=details
        )
    
    @staticmethod
    def print_summary(metrics: EvaluationMetrics, dataset_name: str = None) -> None:
        """Print evaluation summary with detailed logging"""
        dataset_info = f" ({dataset_name})" if dataset_name else ""
        
        print(f"\n{'='*70}")
        print(f"📊 EVALUATION RESULTS{dataset_info}")
        print(f"{'='*70}")
        print(f"  Total Questions:    {metrics.total_questions}")
        print(f"  Correct Answers:    {metrics.correct_answers} ✓")
        print(f"  Incorrect Answers:  {metrics.incorrect_answers} ✗")
        print(f"  Accuracy:           {metrics.accuracy:.1%} ({metrics.accuracy*100:.2f}%)")
        
        # Calculate and display accuracy metrics
        if metrics.details:
            correct_rate = (metrics.correct_answers / metrics.total_questions) * 100 if metrics.total_questions > 0 else 0
            error_rate = (metrics.incorrect_answers / metrics.total_questions) * 100 if metrics.total_questions > 0 else 0
            
            print(f"\n  Performance Metrics:")
            print(f"    • Success Rate:     {correct_rate:.2f}%")
            print(f"    • Error Rate:       {error_rate:.2f}%")
            print(f"    • Questions/sec:    {metrics.total_questions} questions processed")
        
        print(f"{'='*70}\n")
    
    @staticmethod
    def save_results(metrics: EvaluationMetrics, output_file: str) -> None:
        """Save evaluation results to file"""
        results = {
            'accuracy': metrics.accuracy,
            'total_questions': metrics.total_questions,
            'correct_answers': metrics.correct_answers,
            'incorrect_answers': metrics.incorrect_answers,
            'details': metrics.details
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
