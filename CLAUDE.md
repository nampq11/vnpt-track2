# VNPT Track 2 - Vietnamese QA Agent

## Purpose
LLM-based agent for Vietnamese multiple-choice question answering. Built for VNPT Hackathon Track 2.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (use `uv run`, `uv add`, `uv sync`)
- **LLM Backend**: Ollama (default) or Azure OpenAI
- **Default Model**: `qwen3:1.7b`

## Project Structure
```
src/brain/
├── inference/      # Pipeline, processor, evaluator
├── llm/services/   # LLM providers (Ollama, Azure)
├── system_prompt/  # Prompt generation
├── tools/          # Tool management
└── config.py       # Configuration classes
data/               # QA datasets (val.json, test.json)
results/            # Prediction outputs
```

## Quick Commands
```bash
# Install dependencies
uv sync --group development

# Quick test (5 questions)
./bin/inference.sh test

# Full evaluation
./bin/inference.sh eval val

# Inference only (no metrics)
./bin/inference.sh inference test

# Direct CLI
uv run python predict.py --mode eval --input data/val.json --model qwen3:1.7b
```

## Data Format
Questions in `data/*.json`:
- `qid`: Question ID
- `question`: Vietnamese question text (may include context)
- `choices`: Array of 4 options
- `answer`: Letter (A/B/C/D)

## Key Interfaces
- `LLMService` → `src/brain/llm/services/type.py` (abstract base)
- `InferencePipeline` → `src/brain/inference/pipeline.py` (main entry)
- `QuestionProcessor` → `src/brain/inference/processor.py` (parsing)
- `Evaluator` → `src/brain/inference/evaluator.py` (metrics)
- `Config` → `src/brain/config.py` (env-based config)

## Conventions
- Async-first: Use `async/await` for LLM calls
- Type hints required on all public functions
- Vietnamese text handling: Ensure UTF-8 encoding


- After implement code, don't create new any files documentation. only update existing files.
