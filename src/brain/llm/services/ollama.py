from typing import Optional
from src.brain.llm.services.type import LLMService, LLMServiceConfig
from openai import OpenAI
import aiohttp
import ollama
from typing import List

class OllamaServiceConfig(LLMServiceConfig):
    def __init__(self, model: str = "qwen3:1.7b"):
        self.provider = "ollama"
        self.model = model


class OllamaService(LLMService):

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        model: str = "qwen3:1.7b",
        max_iterations: int = 5,
    ) -> None:
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.config = OllamaServiceConfig(model=model)
    
    async def generate(
        self,
        user_input: str,
        stream: Optional[bool] = False,
    ) -> str:
        """Generate response from Ollama via OpenAI API"""
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                stream=stream,
            )
            
            if stream:
                result = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        result += chunk.choices[0].delta.content
                return result
            else:
                return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Error generating response from Ollama: {str(e)}")
    
    async def get_embedding(self, session: aiohttp.ClientSession, text: str) -> List[float]:
        try:
            response = ollama.embed(
                model="nomic-embed-text",
                input=text,
            )
            # Return the first (and only) embedding from the response
            embeddings = response['embeddings']
            return embeddings[0] if embeddings else []
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from Ollama: {str(e)}")

    def get_all_tools(self):
        """Return available tools"""
        return {}
    
    def get_config(self) -> LLMServiceConfig:
        """Return service configuration"""
        return self.config

