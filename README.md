# VNPT Track 2 - Vietnamese QA Agent

## Purpose
LLM-based agent for Vietnamese multiple-choice question answering. Built for VNPT Hackathon Track 2.

## Submission Compliance

This repository complies with VNPT Hackathon Track 2 submission requirements:

✅ **README.md Requirements:**
- ✅ **Pipeline Flow** (Section 1): Detailed system architecture with flow diagram
- ✅ **Data Processing** (Section 2): Complete data collection, cleaning, and processing steps
- ✅ **Resource Initialization** (Section 3): Vector Database (LanceDB), Indexing, and all required resources

✅ **Repository Structure:**
- ✅ `predict.py`: Main entry point (reads `/code/private_test.json`, outputs `submission.csv`)
- ✅ `inference.sh`: Bash script orchestrating the complete pipeline
- ✅ `requirements.txt`: All dependencies with specific versions
- ✅ `Dockerfile`: Builds from clean base image with all resources

✅ **Docker Configuration:**
- ✅ Base image: `python:3.11-slim` (CPU-only, lightweight)
- ✅ CUDA support: Available via `nvidia/cuda:12.2.0-devel-ubuntu20.04` if GPU needed
- ✅ Pre-built knowledge base included (~435MB)
- ✅ Entry point: `inference.sh` (runs complete pipeline)

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: `uv` (development) or `pip` (Docker/submission)
- **LLM Backends**: 
  - **VNPT AI API** (primary) - `vnptai-hackathon-small`, `vnptai-hackathon-large`
  - **Ollama** (local dev) - via OpenAI-compatible API
  - **Azure OpenAI** (optional) - for embeddings
- **Default Model**: `vnptai-hackathon-small` (VNPT)
- **Vector Database**: LanceDB (hybrid search: vector + FTS)

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

This section describes the complete data processing pipeline, including data collection, cleaning, and preparation steps required to run the system.

### 2.1 Data Sources

The system processes Vietnamese question-answering datasets from multiple sources:

**Question Datasets:**
- `data/val.json`: Validation dataset (used for testing/validation)
- `data/test.json`: Test dataset (used for final evaluation)
- `data/private_test.json`: Private test set (mounted by BTC during evaluation)

**Knowledge Base Documents:**
- `data/data/`: Source documents organized by category (26 categories total)
  - Examples: `Bac_Ho/`, `Lich_Su_Viet_nam/`, `Van_Hoa_Viet_Nam/`, `Phap_luat_Viet_Nam/`
  - Format: Plain text files with Wikipedia-style content
  - Total: ~26,513 document chunks after processing

### 2.2 Data Format

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

### 2.3 Data Collection

**Knowledge Base Collection:**
1. **Web Crawling**: Documents are collected from Vietnamese Wikipedia and other sources
   ```bash
   # Crawl data from Wikipedia
   ./bin/crawl.sh -u "https://vi.wikipedia.org/wiki/YOUR_TOPIC" -c "category_name"
   ```

2. **Document Organization**: Documents are organized into 26 categories:
   - History: `Lich_Su_Viet_nam/`, `khang_chien_lon/`, `nhan_vat_lich_su_tieu_bieu/`
   - Culture: `Van_Hoa_Viet_Nam/`, `Van_hoa_am_thuc/`, `le_hoi_truyen_thong/`
   - Geography: `Dia_ly_viet_nam/`, `Dia_hinh_Viet_Nam/`, `Dia_dien_du_lich/`
   - Law: `Phap_luat_Viet_Nam/`, `hien_phap/`, `Quyen_nghia_vu/`
   - Politics: `dang_cong_san_viet_nam/`, `nhan_vat_chinh_tri/`, `cong_hoa_xa_hoi_chu_nghia_VN/`
   - And more...

### 2.4 Data Cleaning & Preparation Steps

**1. Question Data Cleaning** (`notebooks/00-prepare-data.ipynb`)
   - Load JSON files using datasets library
   - Inspect data structure and quality
   - Check for missing or malformed entries
   - Verify UTF-8 Vietnamese text encoding
   - Remove duplicate question IDs
   - Validate JSON structure integrity

**2. Text Preprocessing**
   - Normalize Vietnamese Unicode characters (NFC normalization)
   - Preserve diacritics (tones and accents: á, à, ả, ã, ạ)
   - Maintain original Vietnamese grammar and punctuation
   - Handle context passages (separate from questions when needed)
   - Remove extra whitespace and normalize line breaks

**3. Data Validation**
   - Ensure exactly 4 choices per question
   - Verify answer field contains valid letter (A/B/C/D)
   - Check for duplicate question IDs
   - Validate JSON structure integrity
   - Check for empty or null fields

**4. Knowledge Base Document Processing**
   - **Chunking**: Split documents into 512-character chunks with 50-character overlap
   - **Metadata Extraction**: Extract category, title, section, source file
   - **Text Cleaning**: Remove HTML tags, normalize whitespace, preserve Vietnamese encoding
   - **Tokenization**: Use `underthesea` for Vietnamese word segmentation

**5. Dataset Split**
   - Validation set: `val.json` - used for intermediate testing
   - Test set: `test.json` - used for final evaluation
   - Private test: `private_test.json` - used by BTC for final evaluation
   - No overlap between sets

### 2.5 Data Processing Pipeline

The complete data processing workflow:

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
    
    # Process question text
    # (handled automatically by Agent/InferencePipeline)
```

**Processing Scripts:**
- `predict.py`: Main entry point for inference (reads `/code/private_test.json` in Docker)
- `inference.sh`: Bash script that orchestrates the full pipeline
- `src/brain/inference/pipeline.py`: Core inference pipeline with batch processing

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

#### 3.6 Vector Database Initialization (LanceDB)

The system uses **LanceDB** as the vector database for RAG (Retrieval-Augmented Generation). The knowledge base contains 26,513 document chunks across 26 categories.

**Initialization Methods:**

**Method 1: Pre-built Index (Recommended for Submission)**
The Docker image includes a pre-built LanceDB index at `data/embeddings/knowledge/`:
- **Location**: `data/embeddings/knowledge/knowledge.lance/`
- **Size**: ~435MB
- **Contents**: 26,513 chunks with 1536-dim embeddings (Azure text-embedding-ada-002)
- **Indexes**: Vector (cosine similarity) + Full-Text Search (FTS) + Scalar filters
- **No initialization required** - index is ready to use

**Method 2: Build from Scratch (Development)**
To build the index from source documents:

```bash
# Build complete index from data/data/ directory
uv run python -m src.utils.knowledge_manager build \
    --data-dir data/data \
    --output-dir data/embeddings/knowledge \
    --provider azure

# Or use the knowledge.sh script
./bin/knowledge.sh build --data-dir data/data --provider azure
```

**Method 3: Incremental Updates**
To add new documents to existing index:

```bash
# Upsert new documents (incremental)
./bin/knowledge.sh upsert \
    --data-dir data/data/new_category \
    --provider azure

# Check index status
./bin/knowledge.sh info
```

**LanceDB Index Structure:**
```
data/embeddings/knowledge/
├── knowledge.lance/          # LanceDB database directory
│   ├── _version/             # Version metadata
│   ├── data/                 # Vector data files
│   └── indices/              # Index files (vector, FTS)
├── chunks.json               # Chunk metadata (backup)
├── embeddings.npy            # Raw embeddings (backup)
└── metadata.json             # Build metadata
```

**Index Configuration:**
- **Chunk Size**: 512 characters
- **Overlap**: 50 characters
- **Embedding Dimension**: 1536 (Azure) or 1024 (VNPT)
- **Similarity Metric**: Cosine similarity
- **Hybrid Search**: Vector + Full-Text Search (FTS) + Category filtering

**Loading the Index in Code:**
```python
from src.brain.rag.lancedb_retriever import LanceDBRetriever
from src.brain.llm.services.azure import AzureService

# Initialize LLM service for embeddings
llm_service = AzureService(embedding_model="text-embedding-ada-002")

# Load retriever from index directory
retriever = LanceDBRetriever.from_directory(
    index_dir="data/embeddings/knowledge",
    llm_service=llm_service
)

# Use retriever for semantic search
results = await retriever.retrieve(
    query="Hồ Chí Minh sinh năm nào?",
    top_k=5,
    categories_filter=["Bac_Ho", "Lich_Su_Viet_nam"]
)
```

**Verification:**
```bash
# Check if index exists and is valid
ls -lh data/embeddings/knowledge/knowledge.lance/

# Verify metadata
cat data/embeddings/knowledge/metadata.json
```

#### 3.7 Data Files Initialization

Ensure data files are present:

```bash
# Required files
ls -la data/val.json      # Validation dataset
ls -la data/test.json     # Test dataset

# Verify JSON structure
python3 -m json.tool data/val.json | head -50

# Check knowledge base
ls -la data/embeddings/knowledge/
```

#### 3.8 Entry Point Scripts

The system provides two main entry points for running the pipeline:

**1. `predict.py` - Main Entry Point**
- **Location**: Root directory (`/code/predict.py` in Docker)
- **Purpose**: End-to-end inference pipeline
- **Input**: Reads from `/code/private_test.json` (mounted by BTC)
- **Output**: Generates `submission.csv` with format `qid,answer`
- **Usage**:
  ```bash
  python predict.py \
      --mode inference \
      --input /code/private_test.json \
      --output submission.csv \
      --provider vnpt \
      --use-agent \
      --batch-size 6
  ```

**2. `inference.sh` - Submission Script**
- **Location**: Root directory (`/code/inference.sh` in Docker)
- **Purpose**: Bash script that orchestrates the complete pipeline
- **Executed by**: Docker CMD (default container command)
- **Functionality**:
  - Validates input file exists (`/code/private_test.json`)
  - Checks knowledge base availability
  - Runs `predict.py` with optimal settings
  - Verifies output CSV format
- **Usage** (automatically called by Docker):
  ```bash
  bash inference.sh
  ```

**Pipeline Execution Flow:**
```
Docker Container Start
    ↓
inference.sh (CMD)
    ↓
predict.py --mode inference
    ↓
InferencePipeline.run()
    ↓
Agent.process_query() (for each question)
    ↓
submission.csv (output)
```

### Complete Initialization Checklist

**Required for VNPT (Primary):**
- [ ] Python 3.11+ installed
- [ ] `uv` package manager installed (or use `pip` with `requirements.txt`)
- [ ] Project dependencies installed: `uv sync --group development` OR `pip install -r requirements.txt`
- [ ] VNPT API credentials configured in `config/vnpt.json`
- [ ] System prompt file exists: `src/brain/system_prompt/files/system.md`
- [ ] Data files present: `data/val.json` and `data/test.json`
- [ ] Knowledge base initialized: `data/embeddings/knowledge/knowledge.lance/` (pre-built in Docker)

**Optional for Ollama (Local Development):**
- [ ] Ollama service installed
- [ ] Default model pulled: `ollama pull qwen3:1.7b`
- [ ] Ollama server running: `ollama serve` (listening on port 11434)

**For Docker Submission:**
- [ ] Dockerfile builds successfully
- [ ] `predict.py` and `inference.sh` are in root directory
- [ ] Knowledge base is included in image (`data/embeddings/knowledge/`)
- [ ] `requirements.txt` contains all dependencies with versions
- [ ] Container runs and generates `submission.csv` correctly

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

## Library Management

### Requirements File

The project uses `requirements.txt` for dependency management, as required by submission guidelines.

**File Location**: `requirements.txt` (root directory)

**Management:**
- **Auto-generated**: Created from `pyproject.toml` using `uv pip compile`
- **Version Pinning**: All dependencies have specific versions to ensure reproducibility
- **No Conflicts**: Dependencies are resolved to avoid version conflicts

**Key Dependencies:**
- **LLM Services**: `openai`, `azure-ai-inference`, `httpx`, `aiohttp`
- **Vector Database**: `lancedb`, `numpy`
- **Vietnamese NLP**: `underthesea`
- **Data Processing**: `pandas`, `datasets`
- **Utilities**: `loguru`, `python-dotenv`, `beautifulsoup4`

**Installation:**
```bash
# Using pip (for Docker/submission)
pip install --no-cache-dir -r requirements.txt

# Using uv (for development)
uv sync --group development
```

**Verification:**
```bash
# Check all dependencies are installed
pip list | grep -E "(lancedb|openai|underthesea)"

# Verify no conflicts
pip check
```

**Note**: The `requirements.txt` file is automatically maintained and includes all transitive dependencies with pinned versions to ensure consistent builds across environments.

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

## 4. Docker Deployment (Submission)

### 4.1 Building Docker Image

The project includes a Dockerfile for submission to VNPT Hackathon Track 2, compliant with submission guidelines.

**Prerequisites:**
- Docker installed
- No GPU required (lightweight CPU-only image)
- For GPU support: Use CUDA 12.2 base image (see below)

**Build the image:**
```bash
# CPU-only version (default, recommended)
docker build -t vnpt-track2-submission .

# GPU version (if CUDA 12.2 is needed)
# Modify Dockerfile to use: FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
```

**Key Features:**
- **Base Image**: `python:3.11-slim` (lightweight, ~50MB base, CPU-only)
- **CUDA Support**: Available via `nvidia/cuda:12.2.0-devel-ubuntu20.04` if GPU needed (BTC requirement: CUDA 12.2)
- **Pre-built Knowledge Base**: Included (~435MB) at `data/embeddings/knowledge/`
- **Dependencies**: All installed from `requirements.txt` with pinned versions
- **Entry Point**: `inference.sh` script (runs complete pipeline)
- **Total Image Size**: ~1-2GB (CPU version, much smaller than CUDA version)

**Dockerfile Compliance:**
- ✅ Builds from clean base image
- ✅ All resources (knowledge base, dependencies) included in build
- ✅ No external downloads required at runtime
- ✅ Entry point: `inference.sh` (as per submission requirements)

### 4.2 Running Container

**For submission (BTC evaluation):**
```bash
# BTC will mount private_test.json and run:
docker run \
  -v /path/to/private_test.json:/code/private_test.json \
  vnpt-track2-submission

# Or with GPU (if CUDA 12.2 is available):
docker run --gpus all \
  -v /path/to/private_test.json:/code/private_test.json \
  vnpt-track2-submission
```

**Container Execution Flow:**
1. Container starts → Executes `CMD ["bash", "inference.sh"]`
2. `inference.sh` validates input file exists at `/code/private_test.json`
3. Runs `predict.py --mode inference --input /code/private_test.json --output submission.csv`
4. Pipeline processes all questions using Agent with RAG capabilities
5. Generates `submission.csv` in container root directory

**Output Format (BTC Requirement):**
- **File**: `submission.csv`
- **Location**: Container root (`/code/submission.csv`)
- **Format**: CSV with header `qid,answer`
- **Example**:
  ```csv
  qid,answer
  test_0001,A
  test_0002,B
  test_0003,C
  ```

### 4.3 Configuration

**API Credentials:**
The system requires VNPT AI API credentials. In Docker environment:
- Option A: Mount config file: `-v /path/to/config/vnpt.json:/code/config/vnpt.json`
- Option B: Use environment variables (if implemented)

**Knowledge Base:**
- Pre-built LanceDB index included in image at `data/embeddings/knowledge/`
- Size: ~435MB
- Contains 26,513 chunks across 26 categories
- No initialization required on first run

### 4.4 Multi-threading Configuration

BTC allows and recommends 4-8 parallel threads for optimal inference speed.

**Current Configuration:**
- Default batch size: 6 threads (optimal middle ground)
- Configurable via `--batch-size` argument in `predict.py`
- Too many threads (>8) may cause slower inference due to rate limiting

**Performance:**
- Vector search: <50ms
- Hybrid search: <100ms
- Inference with 6 threads: Optimized for BTC evaluation environment

### 4.5 Testing Locally

Before submission, test the Docker image locally:

```bash
# Build image
docker build -t vnpt-track2-submission .

# Test with sample data
docker run \
  -v $(pwd)/data/test.json:/code/private_test.json \
  vnpt-track2-submission

# Verify output
docker run \
  -v $(pwd)/data/test.json:/code/private_test.json \
  -v $(pwd)/results:/code/results \
  vnpt-track2-submission

# Check submission.csv format
cat results/submission.csv | head -5
```

**Expected output:**
```csv
qid,answer
test_0001,A
test_0002,B
test_0003,C
...
```

### 4.6 Submission Checklist

Before submitting to BTC, verify all requirements:

**Repository Requirements:**
- [x] README.md contains Pipeline Flow (Section 1) with diagram
- [x] README.md contains Data Processing (Section 2) with cleaning steps
- [x] README.md contains Resource Initialization (Section 3) with Vector DB setup
- [x] `predict.py` exists and reads from `/code/private_test.json`
- [x] `inference.sh` exists and orchestrates complete pipeline
- [x] `requirements.txt` contains all dependencies with versions
- [x] `Dockerfile` builds from clean base image

**Docker Requirements:**
- [ ] Docker image builds successfully: `docker build -t vnpt-track2-submission .`
- [ ] Container runs without errors: `docker run -v /path/to/test.json:/code/private_test.json vnpt-track2-submission`
- [ ] `submission.csv` is generated correctly at `/code/submission.csv`
- [ ] CSV format matches: `qid,answer` with proper header
- [ ] Knowledge base is accessible (no errors at `data/embeddings/knowledge/`)
- [ ] API credentials are configured (or fail gracefully)
- [ ] Multi-threading works (4-8 threads recommended, default: 6)
- [ ] Image pushed to Docker Hub with correct tag

**Submission Process:**
1. **Build and Test Locally:**
   ```bash
   docker build -t vnpt-track2-submission .
   docker run -v $(pwd)/data/test.json:/code/private_test.json vnpt-track2-submission
   ```

2. **Push to Docker Hub:**
   ```bash
   docker tag vnpt-track2-submission your-username/vnpt-track2-submission:latest
   docker push your-username/vnpt-track2-submission:latest
   ```

3. **Submit to BTC:**
   - GitHub Repository link (public, no edits after submission)
   - Docker Hub image name: `your-username/vnpt-track2-submission:latest`
   - **Deadline**: 23:59 (UTC+7) ngày 19/12/2024

**Important Notes:**
- Repository must be public and not edited after submission deadline
- Docker image must be pushed before submission deadline
- Images pushed after deadline will be considered invalid

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

