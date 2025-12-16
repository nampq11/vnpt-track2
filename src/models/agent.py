from enum import Enum

class ScenarioTask(str, Enum):
    """Agent Scenario Task"""
    MATH = "MATH"
    READING = "READING"
    RAG = "RAG"
    SAFETY = "SAFETY"