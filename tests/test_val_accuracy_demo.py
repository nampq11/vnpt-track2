#!/usr/bin/env python3
"""
Demo: Accuracy calculation when running val.json evaluation
Shows what the user will see when running with val.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.inference.processor import Question, PredictionResult
from src.brain.inference.evaluator import Evaluator


def demo_val_evaluation():
    """Simulate evaluation on val.json with accuracy logging"""
    print("\n" + "="*70)
    print("DEMO: VAL.JSON EVALUATION WITH ACCURACY LOGGING")
    print("="*70)
    print("\nSimulating: uv run python predict.py --mode eval --input data/val.json")
    print()
    
    # Load actual val.json questions
    val_file = Path(__file__).parent.parent / "data" / "val.json"
    
    print(f"Loading questions from {val_file}...")
    with open(val_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first 5 questions for demo
    sample_data = data[:5]
    questions = [
        Question(
            qid=q['qid'],
            question=q['question'][:80] + "...",
            choices=q['choices'],
            answer=q['answer']
        )
        for q in sample_data
    ]
    
    print(f"✓ Loaded {len(questions)} questions from 'val'")
    print(f"  Output file: results/predictions.csv\n")
    
    # Simulate inference with some realistic predictions
    print("Starting inference...")
    mock_predictions = [
        'B',  # val_0001 - correct
        'A',  # val_0002 - correct
        'B',  # val_0003 - correct
        'B',  # val_0004 - incorrect (expected A)
        'C',  # val_0005 - correct
    ]
    
    predictions = [
        {
            'qid': q.qid,
            'predicted_answer': ans
        }
        for q, ans in zip(questions, mock_predictions)
    ]
    
    print("✓ Processing complete\n")
    
    # Save predictions
    print("Predictions saved to results/predictions.csv\n")
    
    # Evaluate
    print("Evaluating predictions...")
    ground_truth = [q.answer for q in questions]
    
    metrics = Evaluator.evaluate(predictions, ground_truth)
    
    # Print summary with dataset name
    Evaluator.print_summary(metrics, dataset_name="val")
    
    # Show detailed results
    print("Detailed Results:")
    print("-" * 70)
    print(f"{'QID':<12} {'Predicted':<12} {'Ground Truth':<12} {'Status':<10}")
    print("-" * 70)
    
    for detail in metrics.details:
        status = "✓ CORRECT" if detail['correct'] else "✗ WRONG"
        print(f"{detail['qid']:<12} {detail['predicted']:<12} {detail['ground_truth']:<12} {status:<10}")
    
    print("-" * 70)
    
    # Performance summary
    print("\nPerformance Summary:")
    print(f"  Dataset:        val")
    print(f"  Total Tested:   {metrics.total_questions}")
    print(f"  Accuracy:       {metrics.accuracy*100:.2f}% ({metrics.correct_answers}/{metrics.total_questions})")
    print(f"  Correct:        {metrics.correct_answers}")
    print(f"  Incorrect:      {metrics.incorrect_answers}")
    print()
    print("✓ Metrics saved to results/predictions_metrics.json")
    print()


def demo_full_val():
    """Show what would happen with full val.json"""
    print("\n" + "="*70)
    print("DEMO: FULL VAL.JSON STATISTICS")
    print("="*70)
    
    val_file = Path(__file__).parent.parent / "data" / "val.json"
    
    with open(val_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nDataset: val.json")
    print(f"Total questions in val.json: {len(data)}")
    print()
    print("When you run:")
    print("  uv run python predict.py --mode eval --input data/val.json --output results/predictions.csv")
    print()
    print("You will see:")
    print("  1. Questions loaded message with count")
    print("  2. Inference progress for all questions")
    print("  3. Detailed accuracy calculation:")
    print("     - Total Questions: N")
    print("     - Correct Answers: N")
    print("     - Incorrect Answers: N")
    print("     - Accuracy: X.XX%")
    print("  4. Performance metrics (Success/Error rates)")
    print("  5. Detailed prediction-by-prediction breakdown")
    print("  6. Metrics saved to JSON file")
    print()


if __name__ == "__main__":
    demo_val_evaluation()
    demo_full_val()
    
    print("="*70)
    print("Demo completed! This shows what users see when running val.json.")
    print("="*70)

