from typing import Optional, List
from src.brain.llm.services.type import LLMService, LLMServiceConfig
from openai import OpenAI
from loguru import logger
import aiohttp
import ollama
from src.brain.llm.services.retry_utils import retry_sync

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
        verbose: bool = False
    ) -> None:
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.config = OllamaServiceConfig(model=model)
    
    async def generate(
        self,
        user_input: str,
        system_message: Optional[str] = None,
        stream: Optional[bool] = False,
        verbose: bool = False
    ) -> str:
        """
        Generate response from Ollama via OpenAI API.
        Note: OpenAI client has built-in retry logic, no additional retries needed.
        """
        try:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_input})
            if verbose or self.verbose:
                logger.info(
                    f"Sending request to Ollama with model={self.model}, messages={messages}"
                )

            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
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
        """Get embedding from Ollama with retry logic (3 retries, exponential backoff)"""
        try:
            return self._get_embedding_with_retry(text)
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from Ollama: {str(e)}")
    
    @retry_sync(
        max_retries=3,
        exceptions=(ConnectionError, TimeoutError, Exception)
    )
    def _get_embedding_with_retry(self, text: str) -> List[float]:
        """Internal method with retry logic for get_embedding()"""
        response = ollama.embed(
            model="nomic-embed-text",
            input=text,
        )
        # Return the first (and only) embedding from the response
        embeddings = response['embeddings']
        return embeddings[0] if embeddings else []

    def get_all_tools(self):
        """Return available tools"""
        return {}
    
    def get_config(self) -> LLMServiceConfig:
        """Return service configuration"""
        return self.config

