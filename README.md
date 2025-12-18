# VNPT Track 2 - Vietnamese QA Agent

## Purpose
LLM-based agent for Vietnamese multiple-choice question answering. Built for VNPT Hackathon Track 2.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (use `uv run`, `uv add`, `uv sync`)
- **LLM Backends**: 
  - **VNPT AI API** (primary) - `vnptai-hackathon-small`, `vnptai-hackathon-large`
  - **Ollama** (local dev) - via OpenAI-compatible API
- **Default Model**: `vnptai-hackathon-small` (VNPT)

## Project Structure
```
src/brain/
├── agent/          # Agent orchestration & query processing
│   └── tasks/      # Task handlers (math, reading, rag)
├── inference/      # Batch inference pipeline & evaluation
├── llm/
│   ├── messages/   # Conversation & context management
│   └── services/   # LLM providers (VNPT, Ollama)
├── system_prompt/  # Prompt generation
└── config.py       # Configuration classes
data/               # QA datasets (val.json, test.json)
config/             # API credentials (vnpt.json)
tests/              # Integration tests
notebooks/          # Data preparation & experiments
```

## Quick Commands

### Inference & Prediction
```bash
# Install dependencies
uv sync --group development

# Quick test on 5 questions (VNPT)
uv run python predict.py --mode test --input data/test.json --provider vnpt --n 5

# Run full inference on test.json with CSV export (VNPT)
uv run python predict.py --mode inference --input data/test.json --output results/test_predictions.csv --provider vnpt

# Run evaluation on validation set (VNPT)
uv run python predict.py --mode eval --input data/val.json --output results/val_predictions.json --provider vnpt

# Use Ollama instead (requires local Ollama service)
uv run python predict.py --mode test --provider ollama --model qwen3:1.7b --n 5

# Start JupyterLab
uv run jupyter lab
```

### Knowledge Base Management (NEW)
```bash
# Crawl data from Wikipedia
./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/YOUR_TOPIC" \
  -c "category_name"

# Update knowledge index (incremental)
./bin/knowledge.sh upsert --data-dir data/data/category_name --provider azure

# Check index status
./bin/knowledge.sh info

# See full guide: docs/CRAWL_AND_INDEX_GUIDE.md
```

## CLI Usage

The `predict.py` script supports multiple modes and providers:

### Command-Line Arguments

```bash
uv run python predict.py [OPTIONS]

Options:
  --mode {test,eval,inference}
                        Running mode:
                        - test: Quick test on N questions with evaluation
                        - eval: Full dataset evaluation with metrics
                        - inference: Prediction only (no evaluation)
                        Default: test

  --input PATH          Input test file path
                        Default: data/test.json

  --output PATH         Output predictions file path (JSON or CSV)
                        Default: results/predictions.json

  --provider {ollama,vnpt}
                        LLM provider to use
                        Default: vnpt

  --model MODEL         Model name (optional, uses provider default)
                        VNPT: vnptai-hackathon-small (default) or vnptai-hackathon-large
                        Ollama: qwen3:1.7b (default) or any Ollama model

  --n N                 Number of questions to test (test mode only)
                        Default: 5
```

### Usage Examples

```bash
# Test mode: Quick test on 5 questions
uv run python predict.py --mode test --input data/val.json --provider vnpt --n 5

# Eval mode: Full evaluation with accuracy metrics (JSON output)
uv run python predict.py --mode eval --input data/val.json --output results/val_results.json --provider vnpt

# Inference mode: Generate predictions without evaluation (CSV output)
uv run python predict.py --mode inference --input data/test.json --output results/test_predictions.csv --provider vnpt

# Use large model for better accuracy
uv run python predict.py --mode inference --input data/test.json --output results/test_large.csv --provider vnpt --model vnptai-hackathon-large

# Use Ollama for local development
uv run python predict.py --mode test --provider ollama --model qwen3:1.7b --n 10
```

### Output Formats

- **JSON format**: Detailed predictions with metadata
  ```json
  [
    {
      "qid": "test_0001",
      "predicted_answer": "A"
    }
  ]
  ```

- **CSV format**: Simple qid,answer format for submission
  ```csv
  qid,answer
  test_0001,A
  test_0002,B
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
│  • VNPTService setup (Primary):                                      │
│    - Load credentials from config/vnpt.json                          │
│    - Configure model: vnptai-hackathon-small/large                   │
│    - Set up authentication headers (Bearer token, token-id/key)      │
│                                                                       │
│  • OllamaService setup (Optional):                                   │
│    - Initialize OpenAI client pointing to Ollama endpoint           │
│    - Configure model: qwen3:1.7b (default)                          │
│    - Set max_iterations: 5 (iteration limit)                        │
│                                                                       │
│  • Both services implement LLMService interface:                     │
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
│  • Call LLMService.generate(formatted_question)                      │
│    - VNPT: Sends to https://api.idg.vnpt.vn with auth headers       │
│    - Ollama: Sends to local Ollama API endpoint                     │
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
5. **LLM Inference**: Send to VNPT AI API (vnptai-hackathon-small/large) or Ollama for answer generation
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

#### 3.2 VNPT AI Service Configuration (Primary)

The system uses VNPT AI API as the primary LLM backend for the hackathon.

**Configuration:**

Create `config/vnpt.json` with your API credentials:

```json
[
  {
    "authorization": "Bearer YOUR_EMBEDDING_TOKEN",
    "tokenId": "YOUR_EMBEDDING_TOKEN_ID",
    "tokenKey": "YOUR_EMBEDDING_TOKEN_KEY",
    "llmApiName": "vnptai-hackathon-embedding"
  },
  {
    "authorization": "Bearer YOUR_SMALL_MODEL_TOKEN",
    "tokenId": "YOUR_SMALL_TOKEN_ID",
    "tokenKey": "YOUR_SMALL_TOKEN_KEY",
    "llmApiName": "vnptai-hackathon-small"
  },
  {
    "authorization": "Bearer YOUR_LARGE_MODEL_TOKEN",
    "tokenId": "YOUR_LARGE_TOKEN_ID",
    "tokenKey": "YOUR_LARGE_TOKEN_KEY",
    "llmApiName": "vnptai-hackathon-large"
  }
]
```

**Available Models:**
- `vnptai-hackathon-small` (default): Faster, good for most questions
- `vnptai-hackathon-large`: Better accuracy for complex questions
- `vnptai-hackathon-embedding`: For RAG/semantic search tasks

**Usage:**

```python
from src.brain.llm.services.vnpt import VNPTService

# Initialize VNPT service
vnpt_service = VNPTService(
    model="vnptai-hackathon-small",
    model_type="small"  # or "large"
)

# Generate response
response = await vnpt_service.generate("Your question here")
```

#### 3.3 Ollama Service Initialization (Optional - for Local Development)

The system also supports Ollama for local development without requiring API credentials.

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

### 3.4 API Retry Logic

All API calls to external services are protected with automatic retry logic to handle transient failures.

#### Retry Configuration

- **Max Retries**: 3 attempts (4 total tries including initial)
- **Strategy**: Exponential backoff with jitter
- **Backoff Times**: ~1s, ~2s, ~4s (±20% randomization to prevent thundering herd)
- **Max Backoff**: 8 seconds per retry
- **Total Max Time**: ~15 seconds for all retries

#### Services with Retry Logic

| Service | Method | Retry Type | Exceptions Handled |
|---------|--------|------------|-------------------|
| VNPTService | `generate()` | Custom (sync) | RequestException, Timeout, ConnectionError |
| VNPTService | `get_embedding()` | Custom (async) | ClientError, TimeoutError, ConnectionError |
| OllamaService | `generate()` | Built-in SDK | Handled by OpenAI client |
| OllamaService | `get_embedding()` | Custom (sync) | ConnectionError, TimeoutError |
| AzureService | `generate()` | Built-in SDK | Handled by Azure client |
| AzureService | `get_embedding()` | Built-in SDK | Handled by Azure client |

#### Example: Retry in Action

```python
from src.brain.llm.services.vnpt import VNPTService

# Retry happens automatically on API failures
service = VNPTService(model="vnptai-hackathon-small")
response = await service.generate("Your question here")
# If API fails, will retry 3 times with exponential backoff
```

#### Agent Timeout Configuration

The Agent combines retry logic with overall timeout protection:

```python
from src.brain.agent.agent import Agent

agent = Agent(llm_service=service)

# Process with 60s timeout (includes all retries)
result = await agent.process_query(
    query="Your question",
    options={"A": "Option A", "B": "Option B"},
    query_id="q1",
    timeout=60.0  # Optional, default is 60 seconds
)
```

**Timeout Behavior:**
- Prevents queries from hanging indefinitely
- Includes time for: embedding, guardrail, classification, task execution, and all retries
- Returns fallback answer if timeout exceeded
- Logged with query_id for tracking

#### Retry Logs

Retry attempts are logged automatically with context:

```
2025-12-18 23:15:42.123 | WARNING | retry_utils:wrapper:54 - _generate_with_retry attempt 1/4 failed: Connection timeout. Retrying in 1.04s...
2025-12-18 23:15:43.167 | WARNING | retry_utils:wrapper:54 - _generate_with_retry attempt 2/4 failed: Connection timeout. Retrying in 2.20s...
2025-12-18 23:15:45.370 | INFO | Successfully generated response on attempt 3
```

#### 3.5 System Prompt Initialization

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

#### 3.5 Message Formatting & Context Manager Initialization

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

#### 3.6 Data Files Initialization

Ensure data files are present:

```bash
# Required files
ls -la data/val.json      # Validation dataset
ls -la data/test.json     # Test dataset

# Verify JSON structure
python3 -m json.tool data/val.json | head -50
```

### Complete Initialization Checklist

**Required for VNPT (Primary):**
- [ ] Python 3.11+ installed
- [ ] `uv` package manager installed
- [ ] Project dependencies installed: `uv sync --group development`
- [ ] VNPT API credentials configured in `config/vnpt.json`
- [ ] System prompt file exists: `src/brain/system_prompt/files/system.md`
- [ ] Data files present: `data/val.json` and `data/test.json`

**Optional for Ollama (Local Development):**
- [ ] Ollama service installed
- [ ] Default model pulled: `ollama pull qwen3:1.7b`
- [ ] Ollama server running: `ollama serve` (listening on port 11434)

**Legacy Components (if using Agent directly):**
- [ ] Message formatter implemented (IMessageFormatter protocol)
- [ ] ContextManager instantiated with formatter and prompt manager

### Troubleshooting Resource Initialization

**VNPT API Issues:**
```bash
# Verify config file exists and has correct format
cat config/vnpt.json | python -m json.tool

# Test VNPT connection (if you have curl)
curl -X POST https://api.idg.vnpt.vn/data-service/v1/chat/completions/vnptai-hackathon-small \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Token-id: YOUR_TOKEN_ID" \
  -H "Token-key: YOUR_TOKEN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"vnptai_hackathon_small","messages":[{"role":"user","content":"Hello"}]}'

# Check for common errors:
# - 401 Unauthorized: Check authorization token
# - 403 Forbidden: Check tokenId and tokenKey
# - 404 Not Found: Check API endpoint and model name
```

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

### Question Format

Questions in `data/*.json`:
```json
{
  "qid": "test_0001",
  "question": "Vietnamese question text",
  "choices": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "A"
}
```

### Knowledge Base

Source documents in `data/data/`:
```
data/data/
├── Bac_Ho/           # Ho Chi Minh topic
├── Lich_Su_Viet_nam/ # Vietnamese history
├── Van_Hoa_Viet_Nam/ # Vietnamese culture
├── Phap_luat_Viet_Nam/ # Vietnamese law
└── ... (26 categories total)
```

Processed into LanceDB index:
- **26,513 chunks** (512 chars, 50 overlap)
- **1536-dim embeddings** (Azure text-embedding-ada-002)
- **Hybrid indexes**: Vector (cosine) + FTS + Scalar

## Key Interfaces

- `Agent` (`src/brain/agent/agent.py`): Main query processing with classification & task routing
- `LLMService` (`src/brain/llm/services/type.py`): Abstract base for LLM providers
- `VNPTService` (`src/brain/llm/services/vnpt.py`): VNPT AI API client with embedding support
- `OllamaService` (`src/brain/llm/services/ollama.py`): Ollama local LLM client
- `InferencePipeline` (`src/brain/inference/pipeline.py`): Batch inference & evaluation
- `ContextManager` (`src/brain/llm/messages/manager.py`): Manages conversation history
- `EnhancedPromptManager` (`src/brain/system_prompt/enhanced_manager.py`): System prompt generation

## Performance

### Accuracy (val_2 dataset)
- **Overall**: ~80% on test samples
- **Math**: Complex reasoning with step-by-step
- **Reading**: Context-based comprehension
- **RAG**: Knowledge retrieval with category filtering

### RAG Performance
- **Vector search**: <50ms
- **Hybrid search**: <100ms
- **Add 100 docs**: ~30 seconds
- **Delete**: <1 second

---

## Development

### Conventions

- **Async-first**: Use `async/await` for LLM calls
- **Type hints**: Required on all public functions
- **Vietnamese text**: UTF-8 encoding, underthesea tokenization
- **Testing**: Unit tests + integration tests

### Adding New Categories

1. Add documents to `data/data/new_category/`
2. Run: `./bin/knowledge.sh upsert --data-dir data/data/new_category --provider azure`
3. Verify: `./bin/knowledge.sh info`

### Updating Documents

1. Delete old: `./bin/knowledge.sh delete --file "path/to/file.txt"`
2. Add new: `./bin/knowledge.sh upsert --data-dir temp_dir --provider azure`

---

## Documentation

### Core System Documentation
- **Agent System**: `src/brain/agent/README.md` - Agent architecture and task routing
- **RAG System**: `docs/RAG_USAGE_GUIDE.md` - RAG usage and configuration
- **CLI Tools**: `bin/README.md`, `bin/QUICKSTART.md` - Command-line tools
- **Migration**: `plans/20251217-migrate-faiss-to-lancedb/` - LanceDB migration guide
- **CoT Prompts**: `docs/COT_PROMPTS.md` - Chain-of-Thought prompting

### Domain-Aware RAG System (NEW)
- **📚 Complete Guide**: [`docs/CRAWL_AND_INDEX_GUIDE.md`](docs/CRAWL_AND_INDEX_GUIDE.md) - Full workflow for crawling & indexing
- **🚀 Quick Start**: [`docs/QUICK_START_CRAWL.md`](docs/QUICK_START_CRAWL.md) - Quick reference cheat sheet
- **💡 Working Example**: [`examples/add_politics_data_example.sh`](examples/add_politics_data_example.sh) - Complete POLITICS domain example
- **🎯 Implementation**: [`plans/20251218-domain-aware-rag/`](plans/20251218-domain-aware-rag/) - Technical implementation details

### Key Features (Domain-Aware RAG)
- ✅ **6 Domains**: LAW, HISTORY, GEOGRAPHY, CULTURE, **POLITICS** (new), GENERAL_KNOWLEDGE
- ✅ **31 Categories**: Mapped to appropriate domains for intelligent filtering
- ✅ **Domain-Specific Retrieval**: Optimized top_k, vector/FTS weights per domain
- ✅ **Smart Filtering**: Priority-based category selection (Domain → Entity → All)
- ✅ **Easy Data Addition**: Crawl from web → Index → Query (3 steps)

