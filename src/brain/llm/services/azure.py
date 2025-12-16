from typing import Optional, List
from src.brain.llm.services.type import LLMService, LLMServiceConfig
from openai import AzureOpenAI
from dotenv import load_dotenv
import aiohttp
import os

load_dotenv()


class AzureServiceConfig(LLMServiceConfig):
    def __init__(self, model: str = "gpt-4.1"):
        self.provider = "azure"
        self.model = model


class AzureService(LLMService):
    """Azure OpenAI Service implementation"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        model: str = "gpt-4.1",
        embedding_model: str = "text-embedding-ada-002",
        max_iterations: int = 5,
    ) -> None:
        """
        Initialize Azure OpenAI Service
        
        Args:
            api_key: Azure API key (defaults to AZURE_API_KEY env var)
            api_version: Azure API version (defaults to AZURE_API_VERSION env var)
            azure_endpoint: Azure endpoint (defaults to AZURE_ENDPOINT env var)
            model: Model name to use for chat completions
            embedding_model: Model name to use for embeddings
            max_iterations: Max iterations for tool use
        """
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.api_version = api_version or os.getenv("AZURE_API_VERSION")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_ENDPOINT")
        self.model = model
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.config = AzureServiceConfig(model=model)
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
    
    async def generate(
        self,
        user_input: str,
        stream: Optional[bool] = False,
    ) -> str:
        """
        Generate response from Azure OpenAI API
        
        Args:
            user_input: User input text
            stream: Whether to stream response
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
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
            raise RuntimeError(f"Error generating response from Azure: {str(e)}")
    
    async def get_embedding(
        self,
        session: aiohttp.ClientSession,
        text: str,
    ) -> List[float]:
        """
        Get embedding from Azure OpenAI API
        
        Args:
            session: aiohttp session (unused, kept for interface compatibility)
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error getting embedding from Azure: {str(e)}")

    def get_all_tools(self):
        """Return available tools"""
        return {}
    
    def get_config(self) -> LLMServiceConfig:
        """Return service configuration"""
        return self.config

