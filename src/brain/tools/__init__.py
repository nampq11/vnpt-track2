from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from typing import Optional
from brain.llm.services.type import LLMService, LLMServiceConfig

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

class AzureServiceConfig(LLMServiceConfig):
    def __init__(self, model: str) -> None:
        self.provider = "azure"
        self.model = model

class AzureService(LLMService):

    def __init__(
        self,
        model: str,
    ) -> None:
        self.openai = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.model = model
        self.config = AzureServiceConfig(model=model)

    async def generate(self, user_input: str, stream: Optional[bool] = False) -> str:
        
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
            raise RuntimeError(f"Error generating response from Azure: {str(e)}")