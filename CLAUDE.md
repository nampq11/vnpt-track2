# VNPT Track 2 - Vietnamese QA Agent

## Purpose
LLM-based agent for Vietnamese multiple-choice question answering. Built for VNPT Hackathon Track 2.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (use `uv run`, `uv add`, `uv sync`)
- **LLM Backend**: Ollama via OpenAI-compatible API
- **Default Model**: `qwen3:1.7b`

## Project Structure
```
src/brain/
├── agent/          # Agent orchestration
├── llm/
│   ├── messages/   # Conversation & context management
│   └── services/   # LLM service abstractions (Ollama)
├── system-prompt/  # System prompt management
└── utils/          # Shared utilities
data/               # QA datasets (val.json, test.json)
notebooks/          # Data preparation & experiments
```

## Quick Commands
```bash
# Install dependencies
uv sync --group development

# Run prediction
uv run python predict.py

# Run inference script
./bin/inference.sh

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
- `LLMService` (`src/brain/llm/services/type.py`): Abstract base for LLM providers
- `ContextManager` (`src/brain/llm/messages/manager.py`): Manages conversation history
- `EnhancedPromptManager` (`src/brain/system-prompt/enhanced-manager.py`): System prompt generation

## Conventions
- Async-first: Use `async/await` for LLM calls
- Type hints required on all public functions
- Vietnamese text handling: Ensure UTF-8 encoding

