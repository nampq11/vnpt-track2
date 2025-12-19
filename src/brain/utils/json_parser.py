"""Shared JSON parsing utilities for LLM responses"""
import json
import re
from typing import Any, Dict, Optional
from loguru import logger


def parse_json_from_llm_response(
    text: str,
    default: Optional[Dict[str, Any]] = None,
    context: str = "",
    query_id: str = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response text.
    
    Args:
        text: LLM response text (may contain JSON embedded)
        default: Default dict to return on parse failure
        context: Context string for logging (e.g., "QueryClassification")
    
    Returns:
        Parsed JSON dict or default dict
    """
    try:
        # Extract JSON using regex (handles embedded JSON in text)
        match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if match:
            data = json.loads(match.group())
            if verbose:
                logger.debug(f"[{context}] Successfully parsed JSON from response")
            return data
        else:
            if verbose:
                logger.warning(f"[{context}] No JSON found in response: {text[:100]}...")
            return default if default is not None else {}
            
    except json.JSONDecodeError as e:
        logger.error(f"[{context}] JSON decode error: {e}")
        return default if default is not None else {}
    except Exception as e:
        logger.error(f"[{context}] Unexpected error parsing JSON: {e}")
        return default if default is not None else {}


def extract_answer_from_response(
    text: str,
    options: Dict[str, str],
    default_answer: str = "A",
    query_id: str = None,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    Extract answer from LLM response (for QA tasks).
    
    Tries multiple strategies:
    1. Parse JSON with "answer" field
    2. Find quoted letter in text (e.g., "A", 'B')
    3. Return first option as fallback
    
    Args:
        text: LLM response text
        options: Available answer options (A/B/C/D)
        default_answer: Fallback answer
    
    Returns:
        Dict with "answer" key
    """
    # Strategy 1: Parse JSON
    parsed = parse_json_from_llm_response(text, context="AnswerExtraction")
    
    if parsed and "answer" in parsed:
        answer = str(parsed["answer"]).upper().strip()
        if answer in options:
            return {"answer": answer}
    
    # Strategy 2: Find quoted letter
    text_upper = text.upper()
    for letter in sorted(options.keys()):
        if f'"{letter}"' in text_upper or f"'{letter}'" in text_upper:
            if verbose:
                logger.info(f"[{query_id}] Found quoted answer: {letter}")
            return {"answer": letter}
    
    # Strategy 3: Fallback to first option or default
    fallback = sorted(options.keys())[0] if options else default_answer
    if verbose:
        logger.warning(f"[{query_id}] Using fallback answer: {fallback}")
    return {"answer": fallback}

