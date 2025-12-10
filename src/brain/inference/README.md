# Inference Module

Simple inference pipeline cho Vietnamese QA task.

## Structure

- **processor.py** - Question loading & formatting
  - `Question` - Data class for QA items
  - `PredictionResult` - Output data class
  - `QuestionProcessor` - Load & format questions

- **evaluator.py** - Metrics & evaluation
  - `EvaluationMetrics` - Metrics result
  - `Evaluator` - Calculate & save metrics

- **pipeline.py** - Main inference engine
  - `InferencePipeline` - Run inference on batch
  - `run_pipeline()` - Async pipeline runner

- **simple_test.py** - Testing utilities
  - `SimpleInferenceTest` - Quick test helpers

## Usage

### Run Pipeline

```python
import asyncio
from brain.inference.pipeline import run_pipeline

# With evaluation
metrics = await run_pipeline(
    test_file="data/test.json",
    output_file="results/predictions.json",
    evaluate=True
)

# Without evaluation
await run_pipeline(
    test_file="data/test.json",
    output_file="results/predictions.json",
    evaluate=False
)
```

### Quick Test

```python
import asyncio
from brain.inference.simple_test import SimpleInferenceTest

# Test first N questions
await SimpleInferenceTest.test_first_n_questions(
    file_path="data/test.json",
    n=5,
    model="qwen3:1.7b"
)
```

### Manual Processing

```python
from brain.inference.processor import QuestionProcessor, Question

# Load questions
processor = QuestionProcessor()
questions = processor.load_questions("data/test.json")

# Format single question
question = questions[0]
prompt = processor.format_for_llm(question)

# Parse answer
response = "Đáp án: A"
answer = processor.parse_answer(response)  # Returns "A"
```

## Data Format

### Input (test.json)
```json
[
  {
    "qid": "test_0001",
    "question": "Câu hỏi...",
    "choices": ["A", "B", "C", "D"],
    "answer": "A"
  }
]
```

### Output (predictions.json)
```json
[
  {
    "qid": "test_0001",
    "predicted_answer": "A",
    "confidence": 0.0
  }
]
```

### Metrics (predictions_metrics.json)
```json
{
  "accuracy": 0.75,
  "total_questions": 100,
  "correct_answers": 75,
  "incorrect_answers": 25,
  "details": [...]
}
```

## Integration

Module này sẽ được integrate với:
- Agent orchestration layer
- Safety & routing system
- RAG pipelines
- Multi-model ensemble

Hiện tại đây chỉ là simple baseline.

