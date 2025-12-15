"""
Submission Manager for VNPT Hackathon Track 2

Manages CSV submission file with constraints:
- Headers: qid, answer
- Maximum 5 submissions per day
- Keep maximum 10 records (rotate old records)
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class SubmissionRecord:
    """Represents a single submission record"""
    qid: str
    answer: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "qid": self.qid,
            "answer": self.answer
        }


class SubmissionManager:
    """Manages CSV submissions with daily quota and record limits"""
    
    MAX_RECORDS = 10
    MAX_DAILY_SUBMISSIONS = 5
    
    def __init__(self, submission_file: str = "submissions.csv"):
        """
        Initialize submission manager
        
        Args:
            submission_file: Path to CSV submission file
        """
        self.submission_file = Path(submission_file)
        self.metadata_file = Path(str(submission_file).replace(".csv", "_metadata.json"))
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure CSV file and metadata file exist"""
        if not self.submission_file.exists():
            # Create new CSV with headers
            with open(self.submission_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
                writer.writeheader()
        
        if not self.metadata_file.exists():
            # Create metadata tracking daily submissions
            self._save_metadata({
                "created_at": datetime.now().isoformat(),
                "daily_submissions": {},
                "submission_history": []
            })
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "created_at": datetime.now().isoformat(),
                "daily_submissions": {},
                "submission_history": []
            }
    
    def _save_metadata(self, metadata: Dict):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _get_today_key(self) -> str:
        """Get today's date key in YYYY-MM-DD format"""
        return datetime.now().strftime("%Y-%m-%d")
    
    def get_daily_submission_count(self) -> int:
        """Get number of submissions made today"""
        metadata = self._load_metadata()
        today = self._get_today_key()
        return metadata.get("daily_submissions", {}).get(today, 0)
    
    def can_submit(self) -> bool:
        """Check if team can make a submission today"""
        return self.get_daily_submission_count() < self.MAX_DAILY_SUBMISSIONS
    
    def get_remaining_submissions(self) -> int:
        """Get remaining submissions available today"""
        return max(0, self.MAX_DAILY_SUBMISSIONS - self.get_daily_submission_count())
    
    def _read_records(self) -> List[Dict[str, str]]:
        """Read all records from CSV"""
        records = []
        try:
            with open(self.submission_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader is not None:
                    records = list(reader)
        except FileNotFoundError:
            pass
        return records
    
    def _write_records(self, records: List[Dict[str, str]]):
        """Write records to CSV"""
        with open(self.submission_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["qid", "answer"])
            writer.writeheader()
            writer.writerows(records)
    
    def add_submission(self, predictions: List[Dict[str, str]]) -> bool:
        """
        Add predictions as a new submission
        
        Args:
            predictions: List of dicts with 'qid' and 'answer' keys
        
        Returns:
            bool: True if submission successful, False if daily quota exceeded
        """
        if not self.can_submit():
            raise RuntimeError(
                f"Daily submission limit reached ({self.MAX_DAILY_SUBMISSIONS}/day). "
                f"Remaining: {self.get_remaining_submissions()}"
            )
        
        # Load current records
        records = self._read_records()
        
        # Add new predictions
        new_records = records + predictions
        
        # Keep only last MAX_RECORDS
        if len(new_records) > self.MAX_RECORDS:
            removed_count = len(new_records) - self.MAX_RECORDS
            new_records = new_records[-self.MAX_RECORDS:]
            print(f"⚠️  Removed {removed_count} oldest records (limit: {self.MAX_RECORDS})")
        
        # Write updated records
        self._write_records(new_records)
        
        # Update metadata
        metadata = self._load_metadata()
        today = self._get_today_key()
        
        current_count = metadata.get("daily_submissions", {}).get(today, 0)
        metadata["daily_submissions"][today] = current_count + 1
        
        metadata["submission_history"].append({
            "timestamp": datetime.now().isoformat(),
            "count": len(predictions),
            "date": today
        })
        
        self._save_metadata(metadata)
        
        print(f"✅ Submission successful!")
        print(f"   Records added: {len(predictions)}")
        print(f"   Total records: {len(new_records)}/{self.MAX_RECORDS}")
        print(f"   Daily submissions: {metadata['daily_submissions'][today]}/{self.MAX_DAILY_SUBMISSIONS}")
        print(f"   Remaining today: {self.get_remaining_submissions()}")
        
        return True
    
    def submit_from_predictions(self, prediction_file: str) -> bool:
        """
        Submit predictions from a JSON/JSONL file
        
        Args:
            prediction_file: Path to predictions file (JSON or JSONL)
        
        Returns:
            bool: True if successful
        """
        predictions = self._load_predictions(prediction_file)
        if not predictions:
            raise ValueError(f"No predictions found in {prediction_file}")
        
        return self.add_submission(predictions)
    
    def _load_predictions(self, pred_file: str) -> List[Dict[str, str]]:
        """Load predictions from file (JSON or JSONL)"""
        pred_path = Path(pred_file)
        if not pred_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_file}")
        
        predictions = []
        
        try:
            with open(pred_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
                # Try JSONL format (one JSON per line)
                if '\n' in content:
                    for line in content.split('\n'):
                        if line.strip():
                            try:
                                pred = json.loads(line)
                                predictions.append({
                                    "qid": pred.get("qid", ""),
                                    "answer": pred.get("answer", "")
                                })
                            except json.JSONDecodeError:
                                continue
                else:
                    # Try JSON array format
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            predictions.append({
                                "qid": item.get("qid", ""),
                                "answer": item.get("answer", "")
                            })
                    elif isinstance(data, dict):
                        predictions.append({
                            "qid": data.get("qid", ""),
                            "answer": data.get("answer", "")
                        })
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {pred_file}: {e}")
        
        return predictions
    
    def get_submission_status(self) -> Dict:
        """Get current submission status"""
        records = self._read_records()
        today = self._get_today_key()
        metadata = self._load_metadata()
        
        return {
            "total_records": len(records),
            "max_records": self.MAX_RECORDS,
            "daily_submissions_today": self.get_daily_submission_count(),
            "max_daily_submissions": self.MAX_DAILY_SUBMISSIONS,
            "remaining_today": self.get_remaining_submissions(),
            "can_submit": self.can_submit(),
            "today": today,
            "last_submission": metadata["submission_history"][-1] if metadata["submission_history"] else None
        }
    
    def show_status(self):
        """Display submission status"""
        status = self.get_submission_status()
        
        print("\n" + "="*60)
        print("📊 SUBMISSION STATUS")
        print("="*60)
        print(f"Date: {status['today']}")
        print(f"\nRecords in CSV:")
        print(f"  Total: {status['total_records']}/{status['max_records']}")
        print(f"  Status: {'⚠️  FULL' if status['total_records'] >= status['max_records'] else '✅ OK'}")
        print(f"\nDaily Submissions:")
        print(f"  Used: {status['daily_submissions_today']}/{status['max_daily_submissions']}")
        print(f"  Remaining: {status['remaining_today']}")
        print(f"  Can Submit: {'✅ YES' if status['can_submit'] else '❌ NO'}")
        
        if status['last_submission']:
            print(f"\nLast Submission:")
            print(f"  Time: {status['last_submission']['timestamp']}")
            print(f"  Records: {status['last_submission']['count']}")
        print("="*60 + "\n")
    
    def get_records(self) -> List[Dict[str, str]]:
        """Get all current records"""
        return self._read_records()
    
    def clear_records(self):
        """Clear all records (for testing)"""
        self._write_records([])
        print("✅ All records cleared")
    
    def reset_daily_quota(self):
        """Reset daily submission quota (for testing/manual reset)"""
        metadata = self._load_metadata()
        today = self._get_today_key()
        if today in metadata.get("daily_submissions", {}):
            del metadata["daily_submissions"][today]
        self._save_metadata(metadata)
        print(f"✅ Daily quota reset for {today}")


def main():
    """Example usage"""
    import sys
    
    manager = SubmissionManager("submissions.csv")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            manager.show_status()
        elif sys.argv[1] == "submit" and len(sys.argv) > 2:
            try:
                manager.submit_from_predictions(sys.argv[2])
            except Exception as e:
                print(f"❌ Error: {e}")
        elif sys.argv[1] == "clear":
            manager.clear_records()
        elif sys.argv[1] == "reset":
            manager.reset_daily_quota()
        else:
            print("Usage:")
            print("  status               - Show submission status")
            print("  submit <file>        - Submit predictions from file")
            print("  clear                - Clear all records")
            print("  reset                - Reset daily quota")
    else:
        manager.show_status()


if __name__ == "__main__":
    main()

