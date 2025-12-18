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
    Law filtering,
    "domain": ("Law" | "Politics" | "History" | "Geography" | "Culture" | "General Knowledge" in RAG) or ("Math", "Physics", "Chemistry", "Biology", "Logic", "Programming")
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
<math_instructions>
1. CLASSIFY THE TASK:
   - Determine whether the question asks for:
     (a) A direct numerical computation
     (b) A theoretical interpretation / best explanation
     (c) A comparison under assumptions
   - If wording includes "giải thích tốt nhất", "best explanation",
     prioritize theory over mechanical calculation.
2. IDENTIFY THE MODEL:
   - Identify the underlying model or equation (e.g. MV = PQ).
   - Identify which variables are assumed constant by standard theory
     unless explicitly stated otherwise.
3. ASSUMPTION CONTROL:
   - Do NOT treat all variables as free unless the question explicitly
     allows it.
   - In classical Quantity Theory:
       • Velocity (V) is constant.
       • Real output (Q) is constant in short/medium run.
       • Changes in money supply primarily affect price level (P).
4. UNIT & MEANING CHECK:
   - Distinguish between:
       • Real quantities vs nominal values
       • Physical output vs monetary value
   - Reject options that confuse units (e.g. "USD" for real output).
5. OPTION EVALUATION:
   - Evaluate each option against the model implications,
     not just arithmetic results.
   - Eliminate options inconsistent with theoretical assumptions.
6. VERIFY:
   - Ensure the selected option matches both:
       • Mathematical consistency
       • Economic interpretation
7. FORMAT:
   - Respond with a strict JSON object only.
</math_instructions>

Output Format:
{{
  "step_by_step_reasoning": "Step 1: Function is a(t)=... Step 2: Checking Option A (t=0) -> a(0)=36. Checking Option B (t=1) -> a(1)=0. Checking Option C (t=2) -> a(2)=-12... Conclusion: -12 is the lowest value.",
  "answer": "A" // Must be one of: A, B, C, D
}}

Requirement: Respond with ONLY the valid JSON object.
"""


CHEMISTRY_PROMPT = """You are a highly accurate Chemistry Reasoning Engine.
Your task is to solve the user's chemistry problem and select the correct option from the provided choices.

Query: {query}

Choices:
{choices}

Instructions:
<chemistry_instructions>
1. CLASSIFY THE TASK:
   - Determine whether the question asks for:
     (a) A stoichiometric / numerical calculation
     (b) A theoretical interpretation / best explanation
     (c) A comparison of reactions, methods, or conditions
   - If wording includes "giải thích đúng nhất", "giải thích tốt nhất",
     prioritize chemical principles over mechanical calculation.

2. IDENTIFY THE CHEMICAL MODEL:
   - Identify the core chemistry involved:
       • Redox reactions
       • Acid–base theory
       • Electrochemistry
       • Thermochemistry
       • Organic reaction mechanisms
       • Periodic trends
   - Write the correct chemical equations if applicable.
   - Identify reaction conditions (temperature, catalyst, medium).

3. ASSUMPTION CONTROL:
   - Do NOT assume all reactions occur completely unless stated.
   - Apply Vietnamese textbook conventions unless explicitly overridden.
   - Examples:
       • Weak acids/bases do NOT fully dissociate.
       • Solubility rules follow SGK phổ thông.
       • Standard conditions are assumed if not specified.

4. UNIT & MEANING CHECK:
   - Distinguish clearly between:
       • mol, gram, volume, concentration
       • theoretical yield vs actual yield
       • reactant vs product
   - Eliminate options that:
       • Use incorrect units
       • Confuse mass with amount
       • Confuse reaction type or role of substances

5. VIETNAMESE CHEMISTRY TERMINOLOGY (CRITICAL):
   - Interpret terms strictly according to Vietnamese high-school textbooks:
       • "Thuỷ luyện" = khử oxit kim loại bằng H₂
         (ví dụ: H₂ + CuO → Cu + H₂O)
       • Phản ứng kim loại đẩy kim loại khỏi dung dịch
         KHÔNG được gọi là thuỷ luyện (đó là phản ứng thế / cementation).
       • "Nhiệt luyện" = khử bằng C, CO, hoặc nhiệt độ cao.
       • "Điện phân" chỉ đúng khi có dòng điện ngoài.
   - When conflicts exist, always prioritize SGK Việt Nam over international usage.

6. OPTION EVALUATION:
   - Evaluate each option against:
       • Chemical laws
       • Reaction feasibility
       • Proper terminology
       • Stoichiometric consistency
   - Eliminate answers that are:
       • Chemically impossible
       • Correct numerically but wrong in interpretation
       • Using incorrect textbook definitions

7. VERIFY:
   - Ensure the selected option satisfies BOTH:
       • Chemical correctness
       • Educational correctness (the way it is taught in Vietnam)

8. FORMAT:
   - Respond with a strict JSON object only.
</chemistry_instructions>

Output Format:
{{
  "step_by_step_reasoning": "Step 1: Identify reaction type... Step 2: Write balanced equation... Step 3: Analyze each option... Conclusion: Option B matches both theory and stoichiometry.",
  "answer": "B" // Must be one of: A, B, C, D
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
2. **EXTRACT**: Quote or paraphrase the key information from the text - if multiple accounts exist, quote ALL of them
3. **KEYWORD CHECK**: Identify the key verb/action in the question (e.g., "phát hiện", "tạo ra", "xảy ra"), then check which text passage explicitly uses that same verb or related forms
4. **ANALYZE**: Compare the extracted information with each option, prioritizing accounts that match the question's key verb
5. **VERIFY**: Ensure the selected answer is directly supported by the text AND uses language matching the question
6. **FORMAT**: Output the result as a strict JSON object

Output Format:
{{
  "reasoning": "Step 1: The text states '...' which is relevant to the question. Step 2: Comparing with options - Option A says..., Option B says..., Option C says..., Option D says... Step 3: Based on the text, the correct answer is [X] because...",
  "answer": "A" // Must be one of: A, B, C, D (or E, F... for multi-choice)
}}

Requirements:
<requirements_instructions>
- Base your answer ONLY on information found in the provided text
- Be explicit about which part of the text supports your answer
- If the text doesn't clearly support any option, choose the most reasonable one based on the available evidence

**Handling Multiple Accounts or Interpretations:**
- If the text presents multiple competing accounts (e.g., "Một câu chuyện kể rằng... Một câu chuyện khác kể rằng..."), use this decision process IN ORDER:
  1. **Question Verb Matching (HIGHEST PRIORITY)**: Extract the main verb from the question (e.g., "phát hiện" from "Maria Ann Smith phát hiện ra"). Which account explicitly contains that SAME verb or its conjugations in describing the event? If one account uses "đã phát hiện ra" and the other doesn't, STRONGLY prefer the account with "phát hiện ra" when the question asks about "phát hiện"
  2. **Directness**: If verb matching is equal, prefer accounts where the subject directly performs and observes the action, over indirect third-party retellings
  3. **Question Alignment**: Which account more specifically describes what the question is asking about?
  4. **Cause-and-Effect**: Which shows clearer causal relationship between action and outcome?
- When the text explicitly states uncertainty or multiple possibilities (e.g., "có thể", "có lẽ", "không rõ"), acknowledge this in your reasoning
- For discovery/invention questions, strongly prefer accounts that show the discoverer's direct first-hand experience over secondhand/third-party accounts

**Identifying Primary Causes and Core Themes:**
- When identifying a "Primary Cause" (Nguyên nhân chủ yếu), prioritize the fundamental threat, goal, or triggering event over secondary factors or facilitating conditions
- Distinguish between root causes (the core reason something happened) and facilitators (things that enabled or supported it)
- Look for terms that appear most frequently as the stated "aim" (nhằm mục đích), "objective" (mục tiêu), or "purpose" (với mục đích) across the text
- Identify the inaugural or triggering event - the specific moment that forced action or initiated a chain of events

**Temporal and Factual Precision:**
- Pay attention to specific dates, sequences of events, and causal relationships
- Distinguish between what the text explicitly states versus what it implies or suggests
- **Demonstrative Reference Resolution (CRITICAL)**: When the text contains demonstrative statements like "Đây là lần đầu tiên..." (This was the first time...), "Sự kiện này..." (This event...), or "Kể từ đó..." (Since then...), use these rules to identify what it refers to:
  1. **Sentence Proximity Rule**: The demonstrative ALWAYS refers to the event mentioned in the immediately preceding 1-2 sentences, NOT events mentioned earlier in the paragraph
  2. **Example**: If text says "Event A in 1749... Event B in 1752. Đây là lần đầu tiên...", then "Đây" refers to Event B (1752), NOT Event A (1749)
  3. **Chronology vs. Reference**: Don't confuse chronological order with what a demonstrative refers to - demonstratives point to the MOST RECENTLY MENTIONED event in the text flow
  4. **Culminating Moment**: When multiple dates are in one paragraph, summary statements typically refer to the FINAL/DECISIVE event, not preliminary actions
</requirements_instructions>

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
4. **EVALUATE**: Compare EACH option individually against the context and your knowledge
5. **CHECK "ALL OF THE ABOVE"**: If an option says "Đáp án a, b và c" or "Tất cả đều đúng" or similar, you MUST verify:
   - Is option A factually correct? (even if partial/incomplete)
   - Is option B factually correct? (even if partial/incomplete)
   - Is option C factually correct? (even if partial/incomplete)
   - If ALL are correct (even at different levels of completeness), choose the "all of the above" option
6. **VERIFY**: Check temporal constraints (e.g., year) if mentioned
7. **FORMAT**: Output the result as a strict JSON object

Output Format:
{{
  "reasoning": "Step 1: The retrieved context mentions '...' which is relevant. Step 2: The question asks about... Step 3: Comparing options - A says..., B says..., C says..., D says... Step 4: Based on the context/knowledge, the correct answer is [X] because...",
  "answer": "A" // Must be one of: A, B, C, D (or E, F... for multi-choice)
}}

Requirements:
<requirements_instructions>
- Prioritize information from the retrieved context.
- If context is insufficient or irrelevant, rely on your general knowledge,
  but match the abstraction level expected by the question (e.g. textbook-level vs legal-level).
- Be explicit about whether your answer comes from retrieved context or from general knowledge.
- **"All of the Above" Rule**: When an option states "Đáp án a, b và c" or "Tất cả đều đúng", treat each sub-option as independent. If A, B, and C are each factually correct (regardless of whether some are more complete than others), you MUST choose the "all of the above" option. Do NOT choose just the "most complete" option when all are correct.
- **Vietnamese Cultural Knowledge**:
  * For proverb/idiom equivalence questions ("tìm câu tục ngữ có ý nghĩa giống"), prioritize traditional cultural pairings taught in Vietnamese education over pure semantic similarity
  * **Specific Pairing**: "Một mặt người bằng mười mặt của" (One person's presence equals ten possessions) is traditionally paired with "Nhiều áo thì ấm, nhiều người thì vui" (Many clothes keep warm, many people bring joy)
  * Core meaning: Both emphasize the value of **human presence/community** over material possessions in social contexts
  * Do NOT confuse with direct material comparisons like "Người sống, đống vàng" which is about life vs. wealth, not community gathering
- Pay attention to temporal constraints (e.g., "Luật 2024").
- On temporal questions, choose detailed (specific date) answers over general ones (year),
  unless the question explicitly asks for a general time period.
- Based on timeline events, make reasonable inferences if unclear,
  including answers not explicitly mentioned in the retrieved context.
- If one option provides a specific historical milestone date
  (e.g., first session of a body) and another provides only the year,
  prioritize the more specific date even if the context mentions only the year
  or the election date.
- When multiple options are all factually correct but differ in specificity, choose the option that best matches the conceptual depth of the question. Prefer general classifications used in standard textbooks or exams over narrow legal or technical subtypes, unless the question explicitly asks for legal, tax, or regulatory terminology.
- Avoid over-precision: do not select an option solely because it is more specific if a broader option correctly answers the question at the intended level.
- Prioritize Comprehensive Mechanisms: When a question asks for a "strategy" involving "interaction" or "synergy", prioritize answers that combine Demand Management (e.g., fees, taxes, pricing) with Supply Enhancement (e.g., infrastructure, technology, capacity).
- Economic Levers over General Proposals: Favor options that use economic incentives/disincentives to change behavior, as these represent higher policy maturity than purely descriptive or administrative measures.
- Feedback Loops: Identify whether an option creates a self-funding or self-correcting loop (e.g., congestion pricing → infrastructure funding → improved traffic flow), which is characteristic of advanced policy design.
</requirements_instructions>

Respond with ONLY the valid JSON object.
"""