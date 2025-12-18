from enum import Enum

class DomainMathTask(str, Enum):
    """Domain for Math Task"""
    MATH = "Math"
    PHYSICS = "Physics"
    CHEMISTRY = "Chemistry"
    BIOLOGY = "Biology"
    LOGIC = "Logic"
    PROGRAMMING = "Programming"