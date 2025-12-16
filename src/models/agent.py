from enum import Enum

class ScenarioTask(str, Enum):
    """Agent Scenario Task"""
    MATH = "math"
    READING = "reading"
    RAG = "rag"
    SAFETY = "safety"