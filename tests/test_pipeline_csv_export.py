#!/usr/bin/env python3
"""
Test full inference pipeline with CSV export
- Load 5 test questions from val_test_5.json
- Generate mock predictions
- Export to CSV
- Verify CSV format and content
"""

import sys
import csv
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.inference.processor import Question, PredictionResult
from src.brain.inference.pipeline import InferencePipeline
from unittest.mock import MagicMock


def test_csv_pipeline():
    """Test complete pipeline with CSV export"""
    print("\n" + "="*70)
    print("TEST: INFERENCE PIPELINE CSV EXPORT")
    print("="*70)
    
    # Load test data
    test_file = Path(__file__).parent.parent / "data" / "val_test_5.json"
    
    print(f"\n[STEP 1] Loading test data from {test_file.name}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} questions")
    
    # Display questions
    print("\nTest Questions:")
    print("-" * 70)
    for i, q in enumerate(data, 1):
        print(f"{i}. [{q['qid']}] {q['question'][:60]}...")
        print(f"   Expected Answer: {q['answer']}")
    
    # Create mock predictions (simulate LLM inference)
    print("\n[STEP 2] Simulating inference predictions...")
    
    # For testing: use a simple heuristic for predictions
    mock_answers = ['B', 'A', 'B', 'A', 'C']  # Realistic predictions
    
    predictions = []
    for i, (q, answer) in enumerate(zip(data, mock_answers)):
        pred = PredictionResult(
            qid=q['qid'],
            predicted_answer=answer,
            confidence=0.85 + (i * 0.02)  # Varying confidence
        )
        predictions.append(pred)
        print(f"  [{q['qid']}] Predicted: {answer} (confidence: {pred.confidence:.2f})")
    
    # Save to CSV
    output_csv = Path(__file__).parent.parent / "results" / "test_predictions.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[STEP 3] Saving predictions to CSV...")
    dummy_service = MagicMock()
    pipeline = InferencePipeline(llm_service=dummy_service)
    pipeline.save_predictions(predictions, str(output_csv))
    
    # Verify CSV
    print(f"\n[STEP 4] Verifying CSV output...")
    
    assert output_csv.exists(), "❌ CSV file not created"
    print(f"✓ CSV file created: {output_csv}")
    
    # Read and parse CSV
    with open(output_csv, 'r', encoding='utf-8') as f:
        csv_content = f.read()
    
    print(f"\nCSV Content:")
    print("-" * 70)
    print(csv_content)
    print("-" * 70)
    
    # Verify header
    lines = csv_content.strip().split('\n')
    header = lines[0]
    assert header == "qid,answer", f"❌ Header mismatch: {header}"
    print(f"✓ Header correct: {header}")
    
    # Verify data rows
    with open(output_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    assert len(rows) == 5, f"❌ Expected 5 rows, got {len(rows)}"
    print(f"✓ Data rows: {len(rows)}")
    
    # Verify each row
    print(f"\nData Verification:")
    print("-" * 70)
    for i, (row, expected) in enumerate(zip(rows, data)):
        qid = row['qid']
        answer = row['answer']
        expected_qid = expected['qid']
        
        assert qid == expected_qid, f"❌ QID mismatch at row {i}"
        assert answer in ['A', 'B', 'C', 'D'], f"❌ Invalid answer at row {i}: {answer}"
        
        status = "✓"
        print(f"{status} Row {i+1}: {qid} -> {answer}")
    
    # Verify JSON still works (backward compatibility)
    print(f"\n[STEP 5] Testing backward compatibility (JSON export)...")
    output_json = Path(__file__).parent.parent / "results" / "test_predictions.json"
    pipeline.save_predictions(predictions, str(output_json))
    
    assert output_json.exists(), "❌ JSON file not created"
    with open(output_json, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    assert len(json_data) == 5, f"❌ JSON: Expected 5 items, got {len(json_data)}"
    print(f"✓ JSON export works: {output_json}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\nSummary:")
    print(f"  • CSV File: {output_csv}")
    print(f"  • JSON File: {output_json}")
    print(f"  • Questions Processed: {len(predictions)}")
    print(f"  • CSV Format: qid,answer")
    print(f"  • Header: ✓ Correct")
    print(f"  • Data Rows: ✓ 5 rows")
    print(f"  • Backward Compatibility: ✓ JSON works")
    print()


if __name__ == "__main__":
    test_csv_pipeline()

