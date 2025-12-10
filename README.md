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

---

## 1. Pipeline Flow

The Vietnamese QA Agent follows a processing pipeline that transforms user questions into multiple-choice answers using LLM inference.

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT QUESTION                              │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              1. CONTEXT MANAGER INITIALIZATION                       │
│                                                                       │
│  • Initialize ContextManager with:                                   │
│    - Message formatter (protocol: IMessageFormatter)                 │
│    - Prompt manager (EnhancedPromptManager)                          │
│    - History provider (optional: IConversationHistoryProvider)       │
│    - Session ID (for tracking conversation state)                   │
│                                                                       │
│  • Components:                                                        │
│    ├─ InternalMessage: role, content, tool_calls, tool_call_id     │
│    ├─ messages: List[InternalMessage] - maintains conversation      │
│    └─ current_token_count: tracks LLM token usage                   │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              2. SYSTEM PROMPT GENERATION                             │
│                                                                       │
│  • EnhancedPromptManager loads system prompt from:                   │
│    src/brain/system-prompt/files/system.md                          │
│                                                                       │
│  • System prompt provides:                                           │
│    - Agent instructions for Vietnamese QA                           │
│    - Task context (multiple-choice format: A/B/C/D)                 │
│    - Response formatting guidelines                                  │
│    - Tool definitions and usage instructions                        │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              3. MESSAGE FORMATTING & PREPARATION                     │
│                                                                       │
│  • Format user input using IMessageFormatter:                        │
│    - Convert question + choices to standardized format              │
│    - Extract and structure choice options (A, B, C, D)              │
│    - Preserve Vietnamese text encoding (UTF-8)                      │
│                                                                       │
│  • Prepare conversation messages:                                    │
│    - System message: generated system prompt                         │
│    - User message: formatted question + choices                     │
│    - Previous messages: from conversation history (if exists)       │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              4. LLM SERVICE INITIALIZATION                           │
│                                                                       │
│  • OllamaService setup:                                              │
│    - Initialize OpenAI client pointing to Ollama endpoint           │
│    - Configure model: qwen3:1.7b (default)                          │
│    - Set max_iterations: 5 (iteration limit)                        │
│                                                                       │
│  • Ollama Service implements LLMService interface:                   │
│    - generate(user_input: str) -> str: async text generation        │
│    - get_all_tools() -> Tool_Set: retrieve available tools          │
│    - get_config() -> LLMServiceConfig: get service configuration    │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              5. GENERATE ANSWER (LLM INFERENCE)                      │
│                                                                       │
│  • Call OllamaService.generate(formatted_question)                  │
│    - Sends prompt + formatted messages to Ollama API                │
│    - Uses OpenAI-compatible API format                              │
│    - Receives text response with answer and reasoning               │
│                                                                       │
│  • LLM Processing:                                                   │
│    - Context window: manages conversation history                   │
│    - Temperature: controlled via model parameters                   │
│    - Max tokens: limited by model configuration                     │
│    - Tool usage: optional tool calls within response (if enabled)   │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              6. PARSE RESPONSE                                       │
│                                                                       │
│  • MessageFormatter.parse_response(response):                        │
│    - Extract final answer (A/B/C/D) from LLM output                 │
│    - Parse reasoning and confidence if available                    │
│    - Handle streaming responses via parse_stream_response()         │
│                                                                       │
│  • Update conversation context:                                      │
│    - Add assistant response to ContextManager.messages              │
│    - Track any tool calls made by LLM                               │
│    - Save to history_provider (if persistence enabled)              │
│                                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT ANSWER (A/B/C/D)                         │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Processing Steps

1. **Question Input**: Vietnamese multiple-choice question with 4 options
2. **Context Initialization**: Set up conversation manager with formatters and prompt
3. **System Prompt Generation**: Load Vietnamese QA instructions
4. **Message Preparation**: Format question into LLM-compatible format
5. **LLM Inference**: Send to Ollama (qwen3:1.7b) for answer generation
6. **Response Parsing**: Extract answer letter and optional reasoning
7. **Answer Output**: Return final multiple-choice answer (A/B/C/D)

---

## 2. Data Processing

### Data Source

The system processes Vietnamese question-answering datasets stored in `data/` directory:
- `data/val.json`: Validation dataset (used for testing/validation)
- `data/test.json`: Test dataset (used for final evaluation)

### Data Format

Each question in the JSON files follows this structure:

```json
{
  "qid": "val_0001",
  "question": "Vietnamese question text (may include context, references, or passages)",
  "choices": [
    "Option A",
    "Option B",
    "Option C",
    "Option D"
  ],
  "answer": "B"
}
```

**Fields Description:**
- `qid` (string): Unique question identifier
- `question` (string): Vietnamese question text, may contain:
  - Direct questions
  - Reference passages or context
  - Citations and sources
  - Multiple paragraphs of background information
- `choices` (array): Exactly 4 answer options in order [A, B, C, D]
- `answer` (string): Correct answer (A, B, C, or D)

### Data Cleaning & Preparation Steps

The data processing pipeline is managed through:

**1. Data Exploration** (`notebooks/00-prepare-data.ipynb`)
   - Load JSON files using datasets library
   - Inspect data structure and quality
   - Check for missing or malformed entries
   - Verify UTF-8 Vietnamese text encoding

**2. Text Preprocessing**
   - Normalize Vietnamese Unicode characters
   - Preserve diacritics (tones and accents)
   - Maintain original Vietnamese grammar and punctuation
   - Handle context passages (separate from questions when needed)

**3. Data Validation**
   - Ensure 4 choices per question
   - Verify answer field contains valid letter (A/B/C/D)
   - Check for duplicate question IDs
   - Validate JSON structure integrity

**4. Dataset Split**
   - Validation set: `val.json` - used for intermediate testing
   - Test set: `test.json` - used for final evaluation
   - No overlap between sets

### Using Datasets in Notebooks

```python
# Example: Load and inspect data
from datasets import load_dataset

# Load from JSON file
dataset = load_dataset('json', data_files='data/val.json')

# Access data
for sample in dataset['train']:
    qid = sample['qid']
    question = sample['question']
    choices = sample['choices']
    answer = sample['answer']
```

---

## 3. Resource Initialization

### Prerequisites

Before running the pipeline, ensure the following resources are properly initialized:

#### 3.1 Python Environment Setup

```bash
# Install Python 3.11 or later
python3 --version  # Should show Python 3.11+

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync --group development
```

#### 3.2 Ollama Service Initialization

The system requires a running Ollama instance providing an OpenAI-compatible API endpoint.

**Installation & Setup:**

```bash
# 1. Install Ollama (macOS/Linux/Windows)
# Visit: https://ollama.ai or use package manager
# Examples:
#   macOS: brew install ollama
#   Linux: curl -fsSL https://ollama.ai/install.sh | sh
#   Windows: Download from https://ollama.ai/download

# 2. Pull the default model
ollama pull qwen3:1.7b

# 3. Start Ollama service
ollama serve

# The service will be available at: http://localhost:11434
```

**Configuration:**

The `OllamaService` connects via OpenAI-compatible API:

```python
from openai import OpenAI

# Initialize Ollama client
openai_client = OpenAI(
    api_key="ollama",  # Required but unused by Ollama
    base_url="http://localhost:11434/v1"  # Ollama endpoint
)

# Create service instance
from brain.llm.services.ollama import OllamaService

ollama_service = OllamaService(
    openai=openai_client,
    model="qwen3:1.7b",      # Specific model to use
    max_iterations=5          # Max inference iterations
)
```

#### 3.3 System Prompt Initialization

The `EnhancedPromptManager` loads system instructions from:

**File:** `src/brain/system-prompt/files/system.md`

**Initialization:**

```python
from brain.system_prompt.enhanced_manager import EnhancedPromptManager

prompt_manager = EnhancedPromptManager()

# Generate system prompt for Vietnamese QA task
system_prompt = await prompt_manager.generate_system_prompt()
# Returns: SystemPromptResult with content field
```

**System Prompt Contents** should include:
- Agent role and task description for Vietnamese QA
- Instruction to output one answer letter (A/B/C/D)
- Format for reasoning and confidence (if applicable)
- Tool usage guidelines (if tools are enabled)

#### 3.4 Message Formatting & Context Manager Initialization

The pipeline requires implementing the message formatting protocol.

**Components to Initialize:**

```python
from brain.llm.messages.manager import ContextManager, InternalMessage
from brain.system_prompt.enhanced_manager import EnhancedPromptManager

# 1. Create message formatter (implement IMessageFormatter protocol)
# Your formatter should implement:
#   - format(message, context) -> formatted message
#   - parse_response(response) -> parsed answer
#   - parse_stream_response(response) -> stream parsing

# 2. Create prompt manager
prompt_manager = EnhancedPromptManager()

# 3. Initialize context manager
context_manager = ContextManager(
    formatter=your_message_formatter,
    prompt_manager=prompt_manager,
    history_provider=None,  # Optional for persistent history
    session_id="qa_session_001"  # Optional session tracking
)

# System prompt is loaded via:
system_prompt = await context_manager.get_system_prompt()
```

**ContextManager Attributes:**
- `formatter`: Message formatting implementation
- `prompt_manager`: System prompt generator
- `history_provider`: Optional conversation history storage
- `session_id`: Unique identifier for tracking conversation
- `messages`: List[InternalMessage] - conversation history
- `current_token_count`: Tracks LLM token usage

#### 3.5 Data Files Initialization

Ensure data files are present:

```bash
# Required files
ls -la data/val.json      # Validation dataset
ls -la data/test.json     # Test dataset

# Verify JSON structure
python3 -m json.tool data/val.json | head -50
```

### Complete Initialization Checklist

- [ ] Python 3.11+ installed
- [ ] `uv` package manager installed
- [ ] Project dependencies installed: `uv sync --group development`
- [ ] Ollama service installed
- [ ] Default model pulled: `ollama pull qwen3:1.7b`
- [ ] Ollama server running: `ollama serve` (listening on port 11434)
- [ ] OpenAI Python client configured for Ollama endpoint
- [ ] System prompt file exists: `src/brain/system-prompt/files/system.md`
- [ ] Message formatter implemented (IMessageFormatter protocol)
- [ ] ContextManager instantiated with formatter and prompt manager
- [ ] OllamaService initialized with OpenAI client and model
- [ ] Data files present: `data/val.json` and `data/test.json`

### Troubleshooting Resource Initialization

**Ollama Connection Issues:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If connection fails, restart Ollama
ollama serve

# Verify model is available
ollama list
```

**Missing Dependencies:**
```bash
# Reinstall with development dependencies
uv sync --group development --refresh
```

**UTF-8 Encoding Issues:**
```bash
# Ensure UTF-8 locale
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

---

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

