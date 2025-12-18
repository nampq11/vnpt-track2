from enum import Enum

class DomainRAGTask(str, Enum):
    """Domain for RAG Task"""
    LAW = "Law"
    HISTORY = "History"
    GEOGRAPHY = "Geography"
    CULTURE = "Culture"
    POLITICS = "Politics"
    GENERAL_KNOWLEDGE = "General Knowledge"