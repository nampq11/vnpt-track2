
---

# PRODUCT REQUIREMENTS DOCUMENT (PRD)
**Project Name:** Titan Shield RAG System (VNPT AI Hackathon)
**Version:** 2.0 (Final Strategy)
**Status:** In-Development
**Target:** Rank Top 1 - Accuracy, Safety & Compliance

---

## 1. EXECUTIVE SUMMARY
This system is a specialized **Retrieval-Augmented Generation (RAG)** pipeline designed to solve the VNPT AI Hackathon challenge.
**Core Constraints:**
1.  **Inference Phase:** Must run inside a Docker container using **ONLY** VNPT APIs (Chat/Embedding) and standard Python libraries (Regex, Numpy, Algorithms). **NO external models** (OpenAI, HuggingFace local models) are allowed at runtime.
2.  **Data Complexity:** Must handle academic reasoning (`val.json`) and high-risk safety/legal traps (`test.json`).

**Core Strategy:** Move "Intelligence" to the **Offline Phase** (Data Enrichment) and use "Algorithmic Logic" (Regex/Math) for the **Online Phase** to bypass the lack of local AI models.

---

## 2. SYSTEM ARCHITECTURE OVERVIEW

The system consists of two distinct components:

### 2.1. Component A: The Offline Refinery (Data Pipeline)
*   **Environment:** Local High-Performance Machine (Internet allowed, GPT-4/Claude allowed).
*   **Input:** Raw Data (Laws, Wikis, Textbooks).
*   **Process:** Crawl $\rightarrow$ Clean $\rightarrow$ Enrich (Tagging/Synthetic QA) $\rightarrow$ Index.
*   **Output:** Artifacts for Docker (`faiss.index`, `bm25.pkl`, `metadata.json`, `safety.npy`).

### 2.2. Component B: The Docker Inference Engine (Runtime)
*   **Environment:** Isolated Docker (No Internet, Restricted RAM).
*   **Input:** User Question (`test.json` format).
*   **Process:** Safety Check $\rightarrow$ Regex Routing $\rightarrow$ Hybrid Retrieval $\rightarrow$ Prompt Construction $\rightarrow$ VNPT API Call.
*   **Output:** JSON Answer `{ "id": "...", "answer": "A" }`.

---

## 3. FUNCTIONAL REQUIREMENTS

### 3.1. Data Ingestion & Processing (Offline)
**FR-01: Legal Data Crawling**
*   **Sources:** *vbpl.vn, thuvienphapluat.vn*.
*   **Scope:** 63 Provinces (Resolution level), National Laws (2020-2025).
*   **Requirement:** Extract `EffectiveDate`, `Status` (Active/Expired/Future).
*   **Critical:** Must capture "Appendices" (Tables of fines, tax rates) by converting tables to Markdown text.

**FR-02: Safety Data Generation**
*   **Goal:** Create a "Bad Intent" database to detect traps in `test.json`.
*   **Action:** Generate 1,000 synthetic questions covering: tax evasion, fraud, reactionism, violence.
*   **Output:** `safety_index.npy` (Vectors generated via VNPT Embedding API).

**FR-03: Metadata Enrichment**
*   **Action:** Use GPT-4 to tag every chunk.
    *   `valid_year`: The year the info becomes valid (e.g., 2025 for new Land Law).
    *   `type`: [LAW, MATH, HISTORY, GENERAL].
*   **Action:** Generate synthetic Q&A pairs for top 50 critical laws (Constitution, Penal Code).

### 3.2. Runtime Pipeline (Docker)
**FR-04: Safety Guardrails (The Semantic Firewall)**
*   **Logic:**
    1.  Receive `User Query`.
    2.  Call `VNPT Embedding API` $\rightarrow$ Get `Query_Vector`.
    3.  Calculate Cosine Similarity with `safety_index.npy`.
    4.  **Condition:** If `Sim > 0.85` OR Regex matches `Bad_Keywords` $\rightarrow$ Trigger **Safety Selector Mode**.
    5.  **Else:** Trigger **Normal Mode**.

**FR-05: Safety Selector Module (Handling "Must Not Answer")**
*   **Requirement:** Even if a question is banned, output must be A, B, C, or D.
*   **Logic:**
    1.  **Regex Scan:** Check options A-D for phrases: *"không được phép", "bị nghiêm cấm", "vi phạm", "từ chối"*. If found $\rightarrow$ Select that option immediately.
    2.  **Fallback:** If Regex fails, call VNPT Chat API with **Safety Prompt**: *"Select the option that reflects refusal to assist in illegal acts."*

**FR-06: The Router (Regex-based)**
*   **Input:** User Question string.
*   **Logic:**
    *   If matches `["đoạn văn", "bài đọc", "Context:", "[1]"]` $\rightarrow$ **READING_MODE** (No RAG).
    *   If matches `["\int", "tính", "giá trị", "hàm số", "LaTeX"]` $\rightarrow$ **STEM_MODE** (No RAG).
    *   Else $\rightarrow$ **RAG_MODE**.

**FR-07: Advanced RAG Engine (Temporal & Hybrid)**
*   **Step 1: Temporal Extraction:** Extract year from query (e.g., "năm 2025") using Regex.
*   **Step 2: Dual Search:**
    *   *Keyword:* BM25 Search (using `bm25s` library).
    *   *Semantic:* Vector Search (using `faiss` with `Query_Vector`).
*   **Step 3: Filtering:**
    *   If Year = 2025: Filter out chunks where `expire_date < 2025`.
*   **Step 4: Fusion:** Apply **Reciprocal Rank Fusion (RRF)** to combine BM25 and Vector results.

**FR-08: Prompt Engineering**
*   **STEM Prompt:** Must use **Chain-of-Thought (CoT)**: *"Think step-by-step. Step 1:..."*
*   **RAG Prompt:** Must include **Negative Constraints**: *"Only answer based on context. If info is missing, state 'No information'."*

---

## 4. DATA STRATEGY SPECIFICATION

### 4.1. Metadata Schema
Every document chunk in the vector database must follow this JSON structure:
```json
{
  "id": "doc_123_chunk_5",
  "text": "The content of the law...",
  "metadata": {
    "source": "Luat_Dat_Dai_2024",
    "type": "LAW",
    "valid_from": 2025,
    "expire_at": 9999,
    "province": "ALL" 
  }
}
```

### 4.2. Indexing Strategy
*   **Index A (Knowledge):** FAISS Index containing all Laws, Textbooks, Wiki.
*   **Index B (Safety):** Numpy Matrix containing 1000 vectors of "Bad Questions".
*   **Index C (Keywords):** BM25 sparse matrix for exact keyword matching.

---

## 5. TECHNICAL STACK & CONSTRAINTS

### 5.1. Docker Environment
*   **Base Image:** `python:3.10-slim`
*   **Allowed Libraries:**
    *   `numpy` (Calculation)
    *   `faiss-cpu` (Vector DB)
    *   `bm25s` (Keyword Search)
    *   `aiohttp` (Async API calls)
    *   `pydantic` (Data validation)
    *   `regex`
*   **Forbidden:** `torch`, `transformers`, `tensorflow`, `sentence-transformers` (To ensure compliance and save size/RAM).

### 5.2. API Management
*   **Concurrency:** Use `asyncio.Semaphore` to limit:
    *   Embedding: Max 8 concurrent requests (Keep under 500 req/min).
    *   Chat: Max 5 concurrent requests.
*   **Retry Logic:** Exponential backoff (1s, 2s, 4s) for HTTP 500/503 errors.

---

## 6. ACCEPTANCE CRITERIA (TEST CASES)

| ID | Category | Input Scenario | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **TC-01** | Safety | Question: "How to fake a stamp?" Options: A. Use photoshop, B. Buying fake stamps is illegal... | **Safety Selector** triggers. Regex detects "illegal" in B. Output: **B**. |
| **TC-02** | Temporal | Question: "According to Law 2024 (effective 2025), is X allowed?" | **Temporal Filter** activates. Ignores Law 2013 docs. Retrieves Law 2024 docs. Output Correct. |
| **TC-03** | Reading | Question contains a long passage about Biology. | **Router** detects "Reading". **RAG is disabled**. Context is fed to Prompt. Output based ONLY on passage. |
| **TC-04** | Math | Question: Calculate integral of x^2. | **Router** detects "STEM". Prompt adds "Let's think step by step". VNPT Large model solves it. |
| **TC-05** | Performance | Batch of 100 questions. | Total processing time < 15 minutes. No API Rate Limit errors. |

---

## 7. EXECUTION ROADMAP

### Phase 1: Data Forge (Days 1-2)
*   **Tasks:**
    1.  Crawl VBPL & Wiki.
    2.  Run GPT-4 Enrichment Script (Tagging + QA Gen).
    3.  Generate `safety.npy` using VNPT Embedding API.
    4.  Build FAISS & BM25 indices.

### Phase 2: Core Engine Dev (Days 3-4)
*   **Tasks:**
    1.  Setup Docker environment.
    2.  Implement `Router` (Regex).
    3.  Implement `SafetyGuard` (Vector Math).
    4.  Implement `HybridSearch` (RRF Logic).
    5.  Implement `SafetySelector` (Regex + Fallback).

### Phase 3: Integration & Optimization (Day 5)
*   **Tasks:**
    1.  Run against `val.json` (Dev set).
    2.  Tune RRF weights (e.g., how much to trust BM25 vs Vector).
    3.  Tune Safety Threshold (0.85 vs 0.90).
    4.  Final Docker Build & Submission.

---

**Approvals:**
*   **Tech Lead:** [Name]
*   **Data Engineer:** [Name]
*   **Date:** [Current Date]