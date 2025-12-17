#!/usr/bin/env python3
"""Error analysis script for prediction results"""
import json
from collections import Counter
from typing import Dict, List

def analyze_predictions(metrics_file: str):
    """Analyze prediction patterns and errors"""
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    details = data['details']
    
    # Count prediction distribution
    predicted_dist = Counter(d['predicted'] for d in details)
    ground_truth_dist = Counter(d['ground_truth'].upper() for d in details)
    
    # Count errors by predicted answer
    errors_by_predicted = {}
    for choice in ['A', 'B', 'C', 'D', 'E']:
        errors = [d for d in details if d['predicted'] == choice and not d['correct']]
        errors_by_predicted[choice] = len(errors)
    
    # Count correct by predicted answer
    correct_by_predicted = {}
    for choice in ['A', 'B', 'C', 'D', 'E']:
        correct = [d for d in details if d['predicted'] == choice and d['correct']]
        correct_by_predicted[choice] = len(correct)
    
    # Analyze bias
    total_questions = len(details)
    
    print("=" * 70)
    print("🔍 ERROR ANALYSIS REPORT")
    print("=" * 70)
    print(f"\n📊 Overall Results:")
    print(f"  Total Questions: {total_questions}")
    print(f"  Correct: {data['correct_answers']} ({data['accuracy']*100:.2f}%)")
    print(f"  Incorrect: {data['incorrect_answers']} ({(1-data['accuracy'])*100:.2f}%)")
    
    print(f"\n📈 Predicted Answer Distribution:")
    for choice in sorted(predicted_dist.keys()):
        count = predicted_dist[choice]
        pct = (count / total_questions) * 100
        print(f"  {choice}: {count:3d} ({pct:5.2f}%) {'█' * int(pct/2)}")
    
    print(f"\n📉 Ground Truth Distribution:")
    for choice in sorted(ground_truth_dist.keys()):
        count = ground_truth_dist[choice]
        pct = (count / total_questions) * 100
        print(f"  {choice}: {count:3d} ({pct:5.2f}%) {'█' * int(pct/2)}")
    
    print(f"\n🎯 Accuracy by Predicted Answer:")
    for choice in sorted(predicted_dist.keys()):
        total = predicted_dist[choice]
        correct = correct_by_predicted.get(choice, 0)
        incorrect = errors_by_predicted.get(choice, 0)
        if total > 0:
            acc = (correct / total) * 100
            print(f"  {choice}: {correct}/{total} correct ({acc:.2f}%)")
    
    # Detect bias
    print(f"\n⚠️  BIAS DETECTION:")
    max_predicted = max(predicted_dist.values())
    max_choice = [k for k, v in predicted_dist.items() if v == max_predicted][0]
    bias_pct = (max_predicted / total_questions) * 100
    
    if bias_pct > 40:
        print(f"  🚨 STRONG BIAS toward answer '{max_choice}' ({bias_pct:.1f}%)")
        print(f"     Expected: ~{100/len(ground_truth_dist):.1f}% per option")
        print(f"     This suggests systematic prompting or parsing issue!")
    elif bias_pct > 30:
        print(f"  ⚠️  MODERATE BIAS toward answer '{max_choice}' ({bias_pct:.1f}%)")
    else:
        print(f"  ✓ No significant bias detected")
    
    # Sample errors
    print(f"\n📝 Sample Errors (first 10):")
    errors = [d for d in details if not d['correct']][:10]
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err['qid']}: Predicted {err['predicted']}, " 
              f"Actual {err['ground_truth'].upper()}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    import sys
    metrics_file = sys.argv[1] if len(sys.argv) > 1 else "results/predictions_val_metrics.json"
    analyze_predictions(metrics_file)

