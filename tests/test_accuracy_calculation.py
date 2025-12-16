#!/usr/bin/env python3
"""
Test accuracy calculation and logging when running val.json
"""

import sys
import json
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.inference.processor import PredictionResult
from src.brain.inference.evaluator import Evaluator, EvaluationMetrics


def test_accuracy_calculation():
    """Test accuracy calculation with detailed logging"""
    print("\n" + "="*70)
    print("TEST: ACCURACY CALCULATION WITH LOGGING")
    print("="*70)
    
    # Create sample predictions and ground truth
    predictions = [
        {'qid': 'val_0001', 'predicted_answer': 'B'},  # Correct
        {'qid': 'val_0002', 'predicted_answer': 'A'},  # Correct
        {'qid': 'val_0003', 'predicted_answer': 'B'},  # Correct
        {'qid': 'val_0004', 'predicted_answer': 'B'},  # Incorrect (expected A)
        {'qid': 'val_0005', 'predicted_answer': 'C'},  # Correct
    ]
    
    ground_truth = ['B', 'A', 'B', 'A', 'C']
    
    print("\n[STEP 1] Setting up test data...")
    print(f"✓ Predictions: {len(predictions)}")
    print(f"✓ Ground truth: {len(ground_truth)}")
    
    # Run evaluation
    print("\n[STEP 2] Running evaluation...")
    metrics = Evaluator.evaluate(predictions, ground_truth)
    
    # Verify metrics
    print("\n[STEP 3] Verifying metrics...")
    assert metrics.total_questions == 5, f"Expected 5 total, got {metrics.total_questions}"
    print(f"✓ Total questions: {metrics.total_questions}")
    
    assert metrics.correct_answers == 4, f"Expected 4 correct, got {metrics.correct_answers}"
    print(f"✓ Correct answers: {metrics.correct_answers}")
    
    assert metrics.incorrect_answers == 1, f"Expected 1 incorrect, got {metrics.incorrect_answers}"
    print(f"✓ Incorrect answers: {metrics.incorrect_answers}")
    
    expected_accuracy = 4 / 5
    assert abs(metrics.accuracy - expected_accuracy) < 0.001, f"Expected {expected_accuracy}, got {metrics.accuracy}"
    print(f"✓ Accuracy: {metrics.accuracy:.1%} ({metrics.accuracy*100:.2f}%)")
    
    # Test logging output
    print("\n[STEP 4] Testing logging output...")
    print("\nLogged output (from Evaluator.print_summary):")
    print("-" * 70)
    
    # Capture the print output
    old_stdout = sys.stdout
    sys.stdout = output_buffer = StringIO()
    
    Evaluator.print_summary(metrics, dataset_name="val")
    
    sys.stdout = old_stdout
    logged_output = output_buffer.getvalue()
    print(logged_output)
    print("-" * 70)
    
    # Verify logging content
    print("\n[STEP 5] Verifying logging content...")
    assert "EVALUATION RESULTS" in logged_output, "Missing 'EVALUATION RESULTS'"
    print("✓ Contains 'EVALUATION RESULTS'")
    
    assert "val" in logged_output, "Missing dataset name 'val'"
    print("✓ Contains dataset name 'val'")
    
    assert "Total Questions" in logged_output, "Missing 'Total Questions'"
    print("✓ Contains 'Total Questions'")
    
    assert "Correct Answers" in logged_output, "Missing 'Correct Answers'"
    print("✓ Contains 'Correct Answers'")
    
    assert "Incorrect Answers" in logged_output, "Missing 'Incorrect Answers'"
    print("✓ Contains 'Incorrect Answers'")
    
    assert "Accuracy" in logged_output, "Missing 'Accuracy'"
    print("✓ Contains 'Accuracy'")
    
    assert "80.00%" in logged_output, "Missing accuracy percentage"
    print("✓ Contains accuracy percentage (80.00%)")
    
    # Test detail verification
    print("\n[STEP 6] Verifying detail tracking...")
    assert metrics.details is not None, "Details not populated"
    assert len(metrics.details) == 5, f"Expected 5 details, got {len(metrics.details)}"
    print(f"✓ Detail records: {len(metrics.details)}")
    
    # Check first correct prediction
    assert metrics.details[0]['correct'] == True, "First prediction should be correct"
    assert metrics.details[0]['predicted'] == 'B', "First prediction should be B"
    assert metrics.details[0]['ground_truth'] == 'B', "First ground truth should be B"
    print(f"✓ Detail 1: Correct (predicted: {metrics.details[0]['predicted']}, truth: {metrics.details[0]['ground_truth']})")
    
    # Check incorrect prediction
    assert metrics.details[3]['correct'] == False, "Fourth prediction should be incorrect"
    assert metrics.details[3]['predicted'] == 'B', "Fourth prediction should be B"
    assert metrics.details[3]['ground_truth'] == 'A', "Fourth ground truth should be A"
    print(f"✓ Detail 4: Incorrect (predicted: {metrics.details[3]['predicted']}, truth: {metrics.details[3]['ground_truth']})")
    
    print("\n" + "="*70)
    print("✅ ALL ACCURACY TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print(f"  • Accuracy Calculation: ✓ Correct")
    print(f"  • Accuracy Value: {metrics.accuracy*100:.2f}% (4/5 correct)")
    print(f"  • Logging Output: ✓ Detailed and formatted")
    print(f"  • Dataset Detection: ✓ 'val' detected")
    print(f"  • Detail Tracking: ✓ All 5 predictions tracked")
    print()


def test_perfect_accuracy():
    """Test perfect accuracy logging"""
    print("\n" + "="*70)
    print("TEST: PERFECT ACCURACY LOGGING")
    print("="*70)
    
    predictions = [
        {'qid': 'q1', 'predicted_answer': 'A'},
        {'qid': 'q2', 'predicted_answer': 'B'},
        {'qid': 'q3', 'predicted_answer': 'C'},
    ]
    ground_truth = ['A', 'B', 'C']
    
    metrics = Evaluator.evaluate(predictions, ground_truth)
    
    assert metrics.accuracy == 1.0, "Should have perfect accuracy"
    assert metrics.correct_answers == 3, "All 3 should be correct"
    assert metrics.incorrect_answers == 0, "Should have 0 incorrect"
    
    print("✓ Perfect accuracy (100%) calculated correctly")
    print()


def test_zero_accuracy():
    """Test zero accuracy logging"""
    print("\n" + "="*70)
    print("TEST: ZERO ACCURACY LOGGING")
    print("="*70)
    
    predictions = [
        {'qid': 'q1', 'predicted_answer': 'A'},
        {'qid': 'q2', 'predicted_answer': 'B'},
        {'qid': 'q3', 'predicted_answer': 'C'},
    ]
    ground_truth = ['D', 'D', 'D']
    
    metrics = Evaluator.evaluate(predictions, ground_truth)
    
    assert metrics.accuracy == 0.0, "Should have zero accuracy"
    assert metrics.correct_answers == 0, "All should be incorrect"
    assert metrics.incorrect_answers == 3, "Should have 3 incorrect"
    
    print("✓ Zero accuracy (0%) calculated correctly")
    print()


if __name__ == "__main__":
    test_accuracy_calculation()
    test_perfect_accuracy()
    test_zero_accuracy()
    
    print("="*70)
    print("✅ ALL ACCURACY TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)

