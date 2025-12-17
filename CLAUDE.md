# VNPT Track 2 - Vietnamese QA Agent

## Purpose
LLM-based agent for Vietnamese multiple-choice question answering. Built for VNPT Hackathon Track 2.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (use `uv run`, `uv add`, `uv sync`)
- **LLM Backend**: VNPT AI API (primary), Ollama (local dev)
- **Default Model**: `vnptai-hackathon-small`

## Project Structure
```
src/brain/
├── agent/          # Agent orchestration & query processing
│   └── tasks/      # Task handlers (math, reading, rag)
├── inference/      # Batch inference pipeline & evaluation
├── llm/services/   # LLM providers (VNPT, Ollama)
├── rag/            # RAG system (LanceDB, document processor)
├── system_prompt/  # Prompt generation
└── config.py       # Configuration classes
data/               # QA datasets (val.json, test.json)
config/             # API credentials (vnpt.json)
tests/              # Integration tests
```

## Quick Commands
```bash
# Install dependencies
uv sync --group development

# Run prediction
uv run python predict.py

# Run inference script
./bin/inference.sh

# Run evaluation (flexible)
./bin/eval.sh --n 10 --provider azure

# Start JupyterLab
uv run jupyter lab
```

## Data Format
Questions in `data/*.json` follow this structure:
- `qid`: Question ID
- `question`: Vietnamese question text (may include context)
- `choices`: Array of 4 options
- `answer`: Letter (A/B/C/D)

## Key Interfaces
- `Agent` (`src/brain/agent/agent.py`): Main query processing with classification & task routing
- `LLMService` (`src/brain/llm/services/type.py`): Abstract base for LLM providers
- `VNPTService` (`src/brain/llm/services/vnpt.py`): VNPT AI API client with embedding support
- `InferencePipeline` (`src/brain/inference/pipeline.py`): Batch inference & evaluation

## Agent Architecture
Query → Guardrail → Classification → Task Execution (Math/Reading/RAG) → Answer

## Conventions
- Async-first: Use `async/await` for LLM calls
- Type hints required on all public functions
- Vietnamese text handling: Ensure UTF-8 encoding
