#!/usr/bin/env python3
"""Test CSV export functionality"""

import sys
import tempfile
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.brain.inference.processor import PredictionResult
from src.brain.inference.pipeline import InferencePipeline


def test_csv_export():
    """Test CSV export with correct header format"""
    # Create sample predictions
    predictions = [
        PredictionResult(qid="Q001", predicted_answer="A", confidence=0.95),
        PredictionResult(qid="Q002", predicted_answer="C", confidence=0.87),
        PredictionResult(qid="Q003", predicted_answer="B", confidence=0.92),
    ]
    
    # Create temporary CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_predictions.csv"
        
        # Create pipeline (with dummy service - not used for saving)
        from unittest.mock import MagicMock
        dummy_service = MagicMock()
        pipeline = InferencePipeline(llm_service=dummy_service)
        
        # Save predictions as CSV
        pipeline.save_predictions(predictions, str(csv_file))
        
        # Verify file was created
        assert csv_file.exists(), "CSV file not created"
        
        # Read and verify CSV content
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Check header
        assert csv_file.read_text().startswith("qid,answer"), "Header mismatch"
        
        # Check data
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        assert rows[0]["qid"] == "Q001", "Q001 mismatch"
        assert rows[0]["answer"] == "A", "Answer A mismatch"
        assert rows[1]["qid"] == "Q002", "Q002 mismatch"
        assert rows[1]["answer"] == "C", "Answer C mismatch"
        assert rows[2]["qid"] == "Q003", "Q003 mismatch"
        assert rows[2]["answer"] == "B", "Answer B mismatch"
        
        print("✓ CSV export test PASSED")
        print(f"✓ Header: qid,answer")
        print(f"✓ Data rows: {len(rows)}")
        print("\nCSV Content:")
        print(csv_file.read_text())


def test_json_export_still_works():
    """Verify JSON export still works (backward compatibility)"""
    import json
    
    predictions = [
        PredictionResult(qid="Q001", predicted_answer="A", confidence=0.95),
        PredictionResult(qid="Q002", predicted_answer="C", confidence=0.87),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "test_predictions.json"
        
        from unittest.mock import MagicMock
        dummy_service = MagicMock()
        pipeline = InferencePipeline(llm_service=dummy_service)
        
        # Save predictions as JSON
        pipeline.save_predictions(predictions, str(json_file))
        
        # Verify file was created and valid JSON
        assert json_file.exists(), "JSON file not created"
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert len(data) == 2, f"Expected 2 items, got {len(data)}"
        assert data[0]["qid"] == "Q001", "Q001 mismatch"
        assert data[0]["predicted_answer"] == "A", "Answer A mismatch"
        
        print("✓ JSON export test PASSED (backward compatible)")


if __name__ == "__main__":
    test_csv_export()
    test_json_export_still_works()
    print("\n" + "="*50)
    print("All tests PASSED!")
    print("="*50)

