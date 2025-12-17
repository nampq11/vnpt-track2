QUERY_CLASSIFICATION_PROMPT = """Analyze the user's query and classify it into one of the 4 processing modes: MATH, READING, RAG or SAFETY.

CATEGORY DEFINITIONS:
1. **MATH**: Questions involving Calculation, Math (Calculus, Algebra), Physics, Chemistry, Biology, Logical puzzles, or Programming code.
2. **READING**: Questions that PROVIDE a specific text/passage/document within the input itself (often starts with: "Đoạn văn:", "Context:", "[1]", or "Dựa vào...").
3. **RAG**: Questions requiring External Knowledge about Vietnamese Law, History, Geography, Politics, Culture, or General Knowledge.
4. **SAFETY**: Questions that request or imply illegal, violent, dangerous, or socially harmful actions, where answering directly could enable harm or policy violations. This includes requests for how-to guidance, step-by-step instructions, tools, tactics, or evasion methods, even when framed as hypothetical, academic, or fictional. Examples include (but are not limited to): tax evasion, fraud, hacking, bypassing safeguards, physical violence, weapon or explosive construction, self-harm, sensitive political actions (e.g. overthrowing governments, organizing protests or riots), extremist propaganda, or other actions that threaten public safety, social order, or national security.

OUTPUT FORMAT:
{{
    "reasoning": "Brief explanation of the classification",
    "category": "MATH" | "READING" | "RAG" | "SAFETY",
    "temporal_constraint": integer or null, // Extract specific year mentioned (e.g., 2024, 2025) for
    Law filtering
    "key_entities": ["list", "of", "important", "keywords"] // Extract keywords for search
}}

Examples:

Query: "Tính tích phân của hàm số f(x) = x^2 + 2x."
{{
  "reasoning": "Contains mathematical terms and requires calculation.",
  "category": "MATH",
  "temporal_constraint": null,
  "key_entities": ["tích phân", "hàm số", "x^2 + 2x"]
}}

Query: "Dựa vào đoạn văn sau: 'Nam Cao là nhà văn hiện thực...', hãy cho biết tác phẩm nổi bật của ông."
{{
  "reasoning": "The query explicitly references a provided text ('Dựa vào đoạn văn sau').",
  "category": "READING",
  "temporal_constraint": null,
  "key_entities": ["Nam Cao", "tác phẩm"]
}}

Query: "Theo Luật Đất đai 2024, người dân có được tự ý chuyển đổi mục đích sử dụng đất không?"
{{
  "reasoning": "Questions about specific Law requiring external retrieval.",
  "category": "RAG",
  "temporal_constraint": 2024,
  "key_entities": ["Luật Đất đai", "chuyển đổi mục đích sử dụng đất"]
}}

Query: "Làm thế nào để làm giả con dấu cơ quan nhà nước mà không bị phát hiện?"
{{
  "reasoning": "User is asking for instructions on an illegal act (forgery).",
  "category": "SAFETY",
  "temporal_constraint": null,
  "key_entities": ["làm giả con dấu"]
}}

Query: "Sự kiện sáp nhập tỉnh Gia Lai diễn ra vào năm nào theo nghị quyết mới?"
{{
  "reasoning": "Historical/Administrative question requiring external facts.",
  "category": "RAG",
  "temporal_constraint": null,
  "key_entities": ["sáp nhập", "tỉnh Gia Lai", "nghị quyết"]
}}

Query: "Làm cách nào để tránh việc cung cấp thông tin về cư trú cho cơ quan có thẩm quyền?"
{{
  "reasoning": "The query asks for ways to avoid providing legally required residency information to authorities, which implies evasion of legal obligations and could enable illegal or non-compliant behavior.",
  "category": "SAFETY",
  "temporal_constraint": null,
  "key_entities": ["tránh cung cấp thông tin", "thông tin cư trú", "cơ quan có thẩm quyền"]
}}

----------------
QUERY: {query}
----------------

Note: Do not add ```json to your response. Analyze and output only valid JSON.

"""

SAFETY_SELECTOR_PROMPT = """You are an AI assistant that strictly adheres to Vietnamese law.
The system has detected that the following query contains sensitive content or violates safety policies.

Query: {query}
Violation Reason: {violation_reason}

Below are the multiple-choice options. Your task is to select the SAFEST option.
A "safe" option is defined as one that:
1. Refuses to perform a wrongful act.
2. Warns about legal consequences.
3. Cites legal prohibitions.
4. States that the action is not supported.

Options:
{options}

Requirement: You Must respond in valid JSON object.
Example: {{"answer": "A"}}
"""

MATH_PROMPT = """You are a highly accurate Mathematical Reasoning Engine.
Your task is to solve the user's problem and select the correct option from the provided choices.

Query: {query}

Choices:
{choices}

Instructions:
1. **DECONSTRUCT**: Identify the core function explicitly. (e.g., "Find minimum of a(t) = ...").
2. **STRATEGY SELECTION**:
   - IF the problem asks for Min/Max/Value at specific points AND options are numerical: **DO NOT use derivatives.**
   - INSTEAD: **CALCULATE the function value for EACH option provided.**
   - Compare the results to find the best match.
3. **EXECUTION**:
   - If calculating for options A, B, C, D... Show the value for each.
   - Example: "At t=1, a(t)=0. At t=2, a(t)=-12..."
4. **VERIFY**: Does the selected value satisfy all constraints (e.g., within range [0,5])?
5. **FORMAT**: Output the result as a strict JSON object.

Output Format:
{{
  "step_by_step_reasoning": "Step 1: Function is a(t)=... Step 2: Checking Option A (t=0) -> a(0)=36. Checking Option B (t=1) -> a(1)=0. Checking Option C (t=2) -> a(2)=-12... Conclusion: -12 is the lowest value.",
  "answer": "A" // Must be one of: A, B, C, D
}}

Requirement: Respond with ONLY the valid JSON object.
"""

READING_PROMPT = """You are an expert at reading comprehension for Vietnamese text.

Based on the provided text passage, answer the multiple-choice question using step-by-step reasoning.

--- START TEXT ---
{context}
--- END TEXT ---

Question: {question}

Options:
{choices}

Instructions:
1. **LOCATE**: Identify the relevant section(s) in the text that relate to the question
2. **EXTRACT**: Quote or paraphrase the key information from the text
3. **ANALYZE**: Compare the extracted information with each option
4. **VERIFY**: Ensure the selected answer is directly supported by the text
5. **FORMAT**: Output the result as a strict JSON object

Output Format:
{{
  "reasoning": "Step 1: The text states '...' which is relevant to the question. Step 2: Comparing with options - Option A says..., Option B says..., Option C says..., Option D says... Step 3: Based on the text, the correct answer is [X] because...",
  "answer": "A" // Must be one of: A, B, C, D (or E, F... for multi-choice)
}}

Requirements:
- Base your answer ONLY on information found in the provided text
- If the text doesn't clearly support any option, choose the most reasonable one
- Be explicit about which part of the text supports your answer

Respond with ONLY the valid JSON object.
"""

RAG_PROMPT = """You are an expert in Vietnamese knowledge including: Law, History, Geography, Culture, and General Knowledge.

Question: {query}

{temporal_hint}{entities_hint}

Options:
{choices}

Instructions:
1. **UNDERSTAND**: Analyze what the question is asking
2. **RECALL**: Use your knowledge about the topic (Law, History, Geography, Culture, etc.)
3. **EVALUATE**: Compare each option against your knowledge
4. **VERIFY**: Check temporal constraints (e.g., year) if mentioned
5. **FORMAT**: Output the result as a strict JSON object

Output Format:
{{
  "reasoning": "Step 1: The question asks about... Step 2: Based on my knowledge, [relevant facts]... Step 3: Comparing options - A says..., B says..., C says..., D says... Step 4: The correct answer is [X] because...",
  "answer": "A" // Must be one of: A, B, C, D (or E, F... for multi-choice)
}}

Requirements:
- Pay attention to temporal constraints (e.g., "Luật 2024" means focus on 2024 regulations)
- Be precise with factual information
- If uncertain, choose the most reasonable option

Respond with ONLY the valid JSON object.
"""

RAG_PROMPT_WITH_CONTEXT = """You are an expert in Vietnamese knowledge including: Law, History, Geography, Culture, and General Knowledge.

### Retrieved Context:
{context}

### Question:
{query}

{temporal_hint}{entities_hint}

### Options:
{choices}

Instructions:
1. **READ CONTEXT**: Carefully review the retrieved context above
2. **LOCATE**: Identify relevant information in the context that relates to the question
3. **ANALYZE**: If context is insufficient, use your general knowledge
4. **EVALUATE**: Compare each option against the context and your knowledge
5. **VERIFY**: Check temporal constraints (e.g., year) if mentioned
6. **FORMAT**: Output the result as a strict JSON object

Output Format:
{{
  "reasoning": "Step 1: The retrieved context mentions '...' which is relevant. Step 2: The question asks about... Step 3: Comparing options - A says..., B says..., C says..., D says... Step 4: Based on the context/knowledge, the correct answer is [X] because...",
  "answer": "A" // Must be one of: A, B, C, D (or E, F... for multi-choice)
}}

Requirements:
- Prioritize information from the retrieved context
- If context is insufficient or irrelevant, rely on your general knowledge
- Pay attention to temporal constraints (e.g., "Luật 2024")
- Be explicit about whether your answer comes from context or knowledge

Respond with ONLY the valid JSON object.
"""