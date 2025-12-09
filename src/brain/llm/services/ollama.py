from brain.llm.services.type import LLMService
from openai import OpenAI


class OllamaService(LLMService):

    def __init__(
        self,
        openai: OpenAI,
        model: str = "qwen3:1.7b",
        max_iterations: int = 5,
    ) -> None:
        self.openai = openai
        self.model = model
        self.max_iterations = max_iterations
    
    async def generate(
        user_input: str
    ) -> str:
        pass

