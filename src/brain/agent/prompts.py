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

MATH_PROMPT = """You are a math expert.

Query: {query}

choics:
{choices}

Requirement: You Must respond in valid JSON object.
Example: {{"answer": "A"}}
"""