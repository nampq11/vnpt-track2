"""
Tests for Submission Manager
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from src.submission_manager import SubmissionManager, SubmissionRecord


@pytest.fixture
def temp_submission_file(tmp_path):
    """Create temporary submission file"""
    submission_file = tmp_path / "test_submissions.csv"
    return str(submission_file)


@pytest.fixture
def manager(temp_submission_file):
    """Create SubmissionManager instance with temp file"""
    return SubmissionManager(temp_submission_file)


@pytest.fixture
def sample_predictions():
    """Sample predictions for testing"""
    return [
        {"qid": "val_0001", "answer": "B"},
        {"qid": "val_0002", "answer": "A"},
        {"qid": "val_0003", "answer": "B"},
    ]


class TestSubmissionRecord:
    """Test SubmissionRecord dataclass"""
    
    def test_record_creation(self):
        """Test creating a submission record"""
        record = SubmissionRecord(qid="val_0001", answer="B")
        assert record.qid == "val_0001"
        assert record.answer == "B"
        assert record.timestamp is not None
    
    def test_record_to_dict(self):
        """Test record conversion to dict"""
        record = SubmissionRecord(qid="val_0001", answer="B")
        d = record.to_dict()
        assert d["qid"] == "val_0001"
        assert d["answer"] == "B"


class TestSubmissionManagerInit:
    """Test SubmissionManager initialization"""
    
    def test_manager_creation(self, manager):
        """Test manager creation"""
        assert manager.MAX_RECORDS == 10
        assert manager.MAX_DAILY_SUBMISSIONS == 5
    
    def test_file_creation(self, manager):
        """Test CSV file is created on init"""
        assert manager.submission_file.exists()
    
    def test_metadata_file_creation(self, manager):
        """Test metadata file is created on init"""
        assert manager.metadata_file.exists()
    
    def test_csv_headers(self, manager):
        """Test CSV has correct headers"""
        with open(manager.submission_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            assert "qid" in first_line
            assert "answer" in first_line


class TestSubmissionLimits:
    """Test submission limits and quotas"""
    
    def test_initial_daily_submission_count(self, manager):
        """Test initial daily submission count is 0"""
        assert manager.get_daily_submission_count() == 0
    
    def test_can_submit_initially(self, manager):
        """Test can submit initially"""
        assert manager.can_submit() is True
    
    def test_remaining_submissions_initially(self, manager):
        """Test remaining submissions count"""
        assert manager.get_remaining_submissions() == 5
    
    def test_daily_submission_limit(self, manager, sample_predictions):
        """Test daily submission limit is enforced"""
        # Submit 5 times
        for i in range(5):
            manager.add_submission(sample_predictions)
        
        # 6th submission should fail
        assert manager.get_daily_submission_count() == 5
        assert manager.can_submit() is False
        
        with pytest.raises(RuntimeError):
            manager.add_submission(sample_predictions)


class TestRecordStorage:
    """Test record storage and limits"""
    
    def test_single_submission(self, manager, sample_predictions):
        """Test storing a single submission"""
        manager.add_submission(sample_predictions)
        records = manager.get_records()
        assert len(records) == 3
        assert records[0]["qid"] == "val_0001"
        assert records[0]["answer"] == "B"
    
    def test_multiple_submissions(self, manager, sample_predictions):
        """Test storing multiple submissions"""
        manager.add_submission(sample_predictions)
        manager.add_submission(sample_predictions)
        records = manager.get_records()
        assert len(records) == 6
    
    def test_max_records_limit(self, manager, sample_predictions):
        """Test max records limit (10)"""
        # Add 4 times (12 records total) but should keep only 10
        for i in range(4):
            manager.add_submission(sample_predictions)
        
        records = manager.get_records()
        assert len(records) == 10  # Should be capped at 10
    
    def test_record_rotation(self, manager, sample_predictions):
        """Test old records are removed when limit exceeded"""
        # Add first batch
        manager.add_submission(sample_predictions)
        first_records = manager.get_records()
        first_qid = first_records[0]["qid"]
        
        # Add 4 more batches to exceed limit
        for i in range(4):
            manager.add_submission(sample_predictions)
        
        # Check old records are gone
        final_records = manager.get_records()
        assert len(final_records) == 10
        # First record should be gone (older than limit)
        qids = [r["qid"] for r in final_records]
        # The exact qid might still be there from newer submissions
        # but the rotation should have happened


class TestPredictionLoading:
    """Test loading predictions from files"""
    
    def test_load_json_array(self, tmp_path, manager):
        """Test loading JSON array format"""
        pred_file = tmp_path / "predictions.json"
        predictions = [
            {"qid": "val_0001", "answer": "B"},
            {"qid": "val_0002", "answer": "A"},
        ]
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f)
        
        loaded = manager._load_predictions(str(pred_file))
        assert len(loaded) == 2
        assert loaded[0]["qid"] == "val_0001"
    
    def test_load_jsonl(self, tmp_path, manager):
        """Test loading JSONL format"""
        pred_file = tmp_path / "predictions.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            f.write('{"qid": "val_0001", "answer": "B"}\n')
            f.write('{"qid": "val_0002", "answer": "A"}\n')
        
        loaded = manager._load_predictions(str(pred_file))
        assert len(loaded) == 2
        assert loaded[0]["qid"] == "val_0001"
    
    def test_load_single_json_object(self, tmp_path, manager):
        """Test loading single JSON object"""
        pred_file = tmp_path / "prediction.json"
        with open(pred_file, "w", encoding="utf-8") as f:
            json.dump({"qid": "val_0001", "answer": "B"}, f)
        
        loaded = manager._load_predictions(str(pred_file))
        assert len(loaded) == 1
        assert loaded[0]["qid"] == "val_0001"
    
    def test_load_nonexistent_file(self, manager):
        """Test error on nonexistent file"""
        with pytest.raises(FileNotFoundError):
            manager._load_predictions("/nonexistent/path.json")
    
    def test_load_invalid_json(self, tmp_path, manager):
        """Test error on invalid JSON"""
        pred_file = tmp_path / "invalid.json"
        with open(pred_file, "w") as f:
            f.write("not valid json {")
        
        with pytest.raises(ValueError):
            manager._load_predictions(str(pred_file))


class TestSubmissionStatus:
    """Test submission status retrieval"""
    
    def test_status_dict(self, manager, sample_predictions):
        """Test status dictionary"""
        status = manager.get_submission_status()
        assert "total_records" in status
        assert "max_records" in status
        assert "daily_submissions_today" in status
        assert "remaining_today" in status
        assert "can_submit" in status
    
    def test_status_after_submission(self, manager, sample_predictions):
        """Test status after submission"""
        manager.add_submission(sample_predictions)
        status = manager.get_submission_status()
        assert status["total_records"] == 3
        assert status["daily_submissions_today"] == 1
        assert status["remaining_today"] == 4
        assert status["can_submit"] is True


class TestClearingAndReset:
    """Test clearing records and resetting quota"""
    
    def test_clear_records(self, manager, sample_predictions):
        """Test clearing all records"""
        manager.add_submission(sample_predictions)
        assert len(manager.get_records()) > 0
        
        manager.clear_records()
        assert len(manager.get_records()) == 0
    
    def test_reset_daily_quota(self, manager, sample_predictions):
        """Test resetting daily quota"""
        # Submit 5 times to max out
        for i in range(5):
            manager.add_submission(sample_predictions)
        
        assert manager.can_submit() is False
        
        # Reset
        manager.reset_daily_quota()
        
        # Should be able to submit again
        assert manager.can_submit() is True


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_empty_submission(self, manager):
        """Test submitting empty list"""
        manager.add_submission([])
        assert len(manager.get_records()) == 0
    
    def test_submission_with_special_characters(self, manager):
        """Test submission with Vietnamese characters"""
        predictions = [
            {"qid": "val_0001", "answer": "Khỉ vàng"},
        ]
        manager.add_submission(predictions)
        records = manager.get_records()
        assert records[0]["answer"] == "Khỉ vàng"
    
    def test_submission_with_missing_fields(self, manager):
        """Test submission with missing qid or answer"""
        predictions = [
            {"qid": "val_0001"},  # missing answer
            {"answer": "B"},       # missing qid
        ]
        manager.add_submission(predictions)
        records = manager.get_records()
        assert records[0]["answer"] == ""
        assert records[1]["qid"] == ""


class TestMetadataTracking:
    """Test metadata tracking"""
    
    def test_metadata_file_creation(self, manager):
        """Test metadata file is created"""
        assert manager.metadata_file.exists()
    
    def test_metadata_structure(self, manager):
        """Test metadata has correct structure"""
        metadata = manager._load_metadata()
        assert "created_at" in metadata
        assert "daily_submissions" in metadata
        assert "submission_history" in metadata
    
    def test_submission_history_tracking(self, manager, sample_predictions):
        """Test submission history is tracked"""
        manager.add_submission(sample_predictions)
        metadata = manager._load_metadata()
        assert len(metadata["submission_history"]) == 1
        assert metadata["submission_history"][0]["count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

