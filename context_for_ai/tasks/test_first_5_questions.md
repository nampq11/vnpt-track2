# Test Suite: First 5 Validation Questions

## Overview
Created comprehensive test suite for the first 5 questions from `data/val.json` covering diverse question types and complexities.

**Status**: ✅ COMPLETED
**Date**: 2024-12-15
**Test File**: `tests/test_first_5_questions.py`
**Results**: 26/26 PASSED

---

## Test Structure

### Test Classes

#### 1. TestQuestion1ReadingComprehension (3 tests)
Tests the first question about laboratory monkeys used in vaccine production.

**Question Details**:
- **QID**: val_0001
- **Type**: Reading Comprehension (long passage)
- **Topic**: Laboratory monkeys and vaccine production in Vietnam
- **Correct Answer**: B (Khỉ vàng - Golden Monkey)
- **Passage Length**: ~2000 characters with detailed content

**Tests**:
- `test_q1_structure()`: Verifies question ID, answer format, and choice count
- `test_q1_choices_present()`: Validates all 4 monkey species options
- `test_q1_answer_is_b()`: Confirms correct answer is B and maps to "Khỉ vàng"

#### 2. TestQuestion2FactualRecall (3 tests)
Tests a simple factual recall question about temple inauguration.

**Question Details**:
- **QID**: val_0002
- **Type**: Factual Recall / Simple Fact
- **Topic**: "Ngôi chùa Ba La Mật được khai dựng vào năm nào?" (Ba La Mat Temple year)
- **Correct Answer**: A (1886)

**Tests**:
- `test_q2_structure()`: Validates exact question text and 4 years
- `test_q2_year_choices()`: Verifies all choices are valid years
- `test_q2_answer_is_a()`: Confirms correct answer is A (1886)

#### 3. TestQuestion3SocialImpact (3 tests)
Tests a social science question about regulatory impact.

**Question Details**:
- **QID**: val_0003
- **Type**: Social/Economic Science
- **Topic**: Impact of regulations on different domains
- **Correct Answer**: B (Kinh tế - Economics)

**Tests**:
- `test_q3_structure()`: Validates choice count and answer
- `test_q3_question_content()`: Confirms question asks about regulations
- `test_q3_choices_are_domains()`: Validates domain options

#### 4. TestQuestion4STEMCalculation (5 tests)
Tests STEM question requiring economics calculation.

**Question Details**:
- **QID**: val_0004
- **Type**: STEM - Economics / Mathematics
- **Topic**: Price elasticity of demand calculation using midpoint formula
- **Correct Answer**: B (-1.0)
- **Calculation**: Midpoint formula with verified result

**Tests**:
- `test_q4_structure()`: Validates question format
- `test_q4_contains_numbers()`: Confirms numerical data present
- `test_q4_choices_are_negative()`: Validates elasticity values
- `test_q4_midpoint_formula_question()`: Confirms formula method
- `test_q4_answer_calculation()`: Verifies calculation = -1.0

#### 5. TestQuestion5STEMFormula (4 tests)
Tests STEM question about physics formula - parallel resistance.

**Question Details**:
- **QID**: val_0005
- **Type**: STEM - Physics
- **Topic**: Equivalent resistance for parallel circuit
- **Correct Answer**: C ((R1 * R2) / (R1 + R2))
- **Special Feature**: 10 choices instead of standard 4

**Tests**:
- `test_q5_structure()`: Validates 10 choices
- `test_q5_question_content()`: Verifies parallel circuit context
- `test_q5_contains_correct_formula()`: Confirms formula at index C
- `test_q5_distractors_present()`: Checks for misconceptions

#### 6. TestIntegrationFirstFiveQuestions (4 tests)
Integration tests covering all 5 questions.

#### 7. TestQuestionValidation (4 tests)
Data quality and encoding validation.

---

## Test Results Summary

✅ **Total Tests**: 26
✅ **Passed**: 26
❌ **Failed**: 0
⏱️ **Execution Time**: 0.28s

---

## Coverage

| Aspect | Coverage |
|--------|----------|
| **Question Types** | 5 different types |
| **Content Length** | Short to long passages |
| **Domains** | Reading, Economics, History, Physics |
| **Languages** | Vietnamese with proper encoding |
| **Calculations** | Verified mathematical formulas |
| **Physics** | Circuit analysis tested |
| **Data Quality** | Empty fields, duplicates checked |

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/test_first_5_questions.py -v

# Run specific test class
uv run pytest tests/test_first_5_questions.py::TestQuestion4STEMCalculation -v

# Run specific test
uv run pytest tests/test_first_5_questions.py::TestQuestion4STEMCalculation::test_q4_answer_calculation -v
```

---

## Conclusion

✅ Comprehensive test suite successfully created and validated for first 5 questions covering:
- Reading comprehension with long passages
- Factual recall with simple facts
- Social science with abstract concepts
- STEM mathematics with calculation verification
- STEM physics with formula validation
- Data quality and Vietnamese encoding

All 26 tests passed, confirming data integrity and format compliance.
