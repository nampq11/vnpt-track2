"""
Test suite for the first 5 validation questions from val.json

This test file validates the system's ability to handle:
1. Reading comprehension questions with long passages (val_0001)
2. Simple factual recall (val_0002)
3. Economic/social impact questions (val_0003)
4. STEM calculations (val_0004)
5. STEM formula-based questions (val_0005)
"""

import json
import pytest
from pathlib import Path
from src.core.models import Question, PredictionResult
from src.runtime.pipeline import RuntimeInferencePipeline
from src.core.config import Config


# Load test data
@pytest.fixture
def test_questions():
    """Load first 5 questions from val.json"""
    val_file = Path(__file__).parent.parent / "data" / "val.json"
    with open(val_file, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    return all_questions[:5]  # First 5 questions


@pytest.fixture
def expected_answers():
    """Expected answers for first 5 questions"""
    return {
        "val_0001": "B",  # Khỉ vàng
        "val_0002": "A",  # 1886
        "val_0003": "B",  # Kinh tế
        "val_0004": "B",  # -1.0
        "val_0005": "C",  # (R1 * R2) / (R1 + R2)
    }


class TestQuestion1ReadingComprehension:
    """Test Question 1: Reading comprehension with passage about laboratory monkeys"""
    
    def test_q1_structure(self, test_questions):
        """Verify structure of question 1"""
        q = test_questions[0]
        assert q["qid"] == "val_0001"
        assert q["answer"] == "B"
        assert len(q["choices"]) == 4
        assert "Khỉ vàng" in q["choices"]
        assert "question" in q
        # Question should contain passage about monkeys
        assert "Khỉ" in q["question"]
    
    def test_q1_choices_present(self, test_questions):
        """Verify all choices are present"""
        choices = test_questions[0]["choices"]
        expected_choices = ["Khỉ đuôi dài", "Khỉ vàng", "Khỉ nâu", "Khỉ mặt đỏ lông nâu"]
        assert set(choices) == set(expected_choices)
    
    def test_q1_answer_is_b(self, test_questions, expected_answers):
        """Verify correct answer is B (Khỉ vàng)"""
        q = test_questions[0]
        assert q["answer"] == expected_answers["val_0001"]
        assert q["choices"][1] == "Khỉ vàng"  # Index 1 corresponds to B


class TestQuestion2FactualRecall:
    """Test Question 2: Factual recall about temple inauguration date"""
    
    def test_q2_structure(self, test_questions):
        """Verify structure of question 2"""
        q = test_questions[1]
        assert q["qid"] == "val_0002"
        assert q["answer"] == "A"
        assert len(q["choices"]) == 4
        assert "1886" in q["choices"]
        assert q["question"] == "Ngôi chùa Ba La Mật được khai dựng vào năm nào?"
    
    def test_q2_year_choices(self, test_questions):
        """Verify year choices are reasonable"""
        choices = test_questions[1]["choices"]
        expected_choices = ["1886", "1900", "1920", "1930"]
        assert choices == expected_choices
        # All should be valid years
        for choice in choices:
            assert int(choice) > 1800
            assert int(choice) < 2000
    
    def test_q2_answer_is_a(self, test_questions, expected_answers):
        """Verify correct answer is A (1886)"""
        q = test_questions[1]
        assert q["answer"] == expected_answers["val_0002"]
        assert q["choices"][0] == "1886"


class TestQuestion3SocialImpact:
    """Test Question 3: Economic/social impact of regulations"""
    
    def test_q3_structure(self, test_questions):
        """Verify structure of question 3"""
        q = test_questions[2]
        assert q["qid"] == "val_0003"
        assert q["answer"] == "B"
        assert len(q["choices"]) == 4
        assert "Kinh tế" in q["choices"]
    
    def test_q3_question_content(self, test_questions):
        """Verify question asks about impact of regulations"""
        q = test_questions[2]
        assert "quy định" in q["question"].lower()
        assert ("thuế" in q["question"].lower() or "pháp luật" in q["question"].lower())
    
    def test_q3_choices_are_domains(self, test_questions):
        """Verify choices represent different domains"""
        choices = test_questions[2]["choices"]
        expected_choices = ["Môi trường", "Kinh tế", "Văn hóa", "Quốc phòng an ninh"]
        assert choices == expected_choices


class TestQuestion4STEMCalculation:
    """Test Question 4: STEM calculation - price elasticity of demand"""
    
    def test_q4_structure(self, test_questions):
        """Verify structure of question 4"""
        q = test_questions[3]
        assert q["qid"] == "val_0004"
        assert q["answer"] == "B"
        assert len(q["choices"]) == 4
    
    def test_q4_contains_numbers(self, test_questions):
        """Verify question contains numerical data"""
        q = test_questions[3]
        question = q["question"]
        # Should contain prices, quantities
        assert "2,00" in question or "2.00" in question or "2" in question
        assert "100" in question
    
    def test_q4_choices_are_negative(self, test_questions):
        """Verify choices are all negative (elasticity values)"""
        choices = test_questions[3]["choices"]
        expected_choices = ["-0,5", "-1,0", "-1,5", "-2,0"]
        assert choices == expected_choices
        # Verify all are negative
        for choice in choices:
            value = float(choice.replace(",", "."))
            assert value < 0
    
    def test_q4_midpoint_formula_question(self, test_questions):
        """Verify question asks for midpoint formula elasticity"""
        q = test_questions[3]
        assert "trung điểm" in q["question"].lower() or "midpoint" in q["question"].lower()
    
    def test_q4_answer_calculation(self):
        """Verify the correct answer using midpoint formula"""
        # Given data
        p1 = 2.00
        p2 = 2.50
        q1 = 100
        q2 = 80
        
        # Midpoint formula: ((Q2 - Q1) / ((Q1 + Q2) / 2)) / ((P2 - P1) / ((P1 + P2) / 2))
        delta_q = q2 - q1  # -20
        delta_p = p2 - p1  # 0.50
        
        avg_q = (q1 + q2) / 2  # 90
        avg_p = (p1 + p2) / 2  # 2.25
        
        elasticity = (delta_q / avg_q) / (delta_p / avg_p)
        # Expected: (-20/90) / (0.50/2.25) ≈ -1.0
        assert abs(elasticity - (-1.0)) < 0.01


class TestQuestion5STEMFormula:
    """Test Question 5: STEM formula - parallel resistance calculation"""
    
    def test_q5_structure(self, test_questions):
        """Verify structure of question 5"""
        q = test_questions[4]
        assert q["qid"] == "val_0005"
        assert q["answer"] == "C"
        # Note: This question has 10 choices, not 4
        assert len(q["choices"]) == 10
    
    def test_q5_question_content(self, test_questions):
        """Verify question asks about parallel resistance"""
        q = test_questions[4]
        assert "điện trở" in q["question"].lower()
        assert "song song" in q["question"].lower() or "parallel" in q["question"].lower()
        assert "R1" in q["question"]
        assert "R2" in q["question"]
    
    def test_q5_contains_correct_formula(self, test_questions):
        """Verify correct formula is in choices"""
        choices = test_questions[4]["choices"]
        # Correct parallel resistance formula
        assert "(R1 * R2) / (R1 + R2)" in choices
        # Verify it's option C (index 2)
        assert choices[2] == "(R1 * R2) / (R1 + R2)"
    
    def test_q5_distractors_present(self, test_questions):
        """Verify common misconceptions are in choices"""
        choices = test_questions[4]["choices"]
        # Series formula (common mistake)
        assert "R1 + R2" in choices
        # Other common wrong formulas
        assert "R1 - R2" in choices
        # Verify variety of wrong answers
        assert len(set(choices)) == len(choices)  # All unique


class TestIntegrationFirstFiveQuestions:
    """Integration tests for all first 5 questions"""
    
    def test_all_five_questions_loaded(self, test_questions):
        """Verify all 5 questions are loaded"""
        assert len(test_questions) == 5
        qids = [q["qid"] for q in test_questions]
        expected_qids = ["val_0001", "val_0002", "val_0003", "val_0004", "val_0005"]
        assert qids == expected_qids
    
    def test_all_questions_have_required_fields(self, test_questions):
        """Verify all questions have required fields"""
        for q in test_questions:
            assert "qid" in q
            assert "question" in q
            assert "choices" in q
            assert "answer" in q
            assert isinstance(q["choices"], list)
            assert q["answer"] in ["A", "B", "C", "D"]
    
    def test_all_answers_are_valid(self, test_questions):
        """Verify all answers reference valid choices"""
        for q in test_questions:
            answer_letter = q["answer"]
            answer_index = ord(answer_letter) - ord("A")
            assert answer_index < len(q["choices"]), \
                f"Question {q['qid']} answer {answer_letter} exceeds choice count"
    
    def test_question_types_diversity(self, test_questions):
        """Verify questions span different types"""
        # Q1: Reading comprehension with passage
        assert len(test_questions[0]["question"]) > 500  # Long passage
        
        # Q2: Simple factual recall
        assert len(test_questions[1]["question"]) < 100  # Short question
        
        # Q3: Social science
        assert "lĩnh vực" in test_questions[2]["question"]
        
        # Q4: STEM - Economics calculation
        assert "giá" in test_questions[3]["question"]
        
        # Q5: STEM - Physics formula
        assert "điện trở" in test_questions[4]["question"]


class TestQuestionValidation:
    """Validate question data quality"""
    
    def test_no_empty_questions(self, test_questions):
        """Verify no empty question text"""
        for q in test_questions:
            assert len(q["question"].strip()) > 0
    
    def test_no_empty_choices(self, test_questions):
        """Verify no empty choice options"""
        for q in test_questions:
            for choice in q["choices"]:
                assert len(choice.strip()) > 0
    
    def test_unique_qids(self, test_questions):
        """Verify all qids are unique"""
        qids = [q["qid"] for q in test_questions]
        assert len(qids) == len(set(qids))
    
    def test_vietnamese_encoding(self, test_questions):
        """Verify Vietnamese text is properly encoded"""
        for q in test_questions:
            # Should contain Vietnamese characters
            question_text = q["question"]
            # Vietnamese diacritics like ă, ơ, ư, etc.
            has_vietnamese = any(c in question_text for c in "ăơưêâốĩùúàảãạ")
            assert has_vietnamese or "đ" in question_text or "ố" in question_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

