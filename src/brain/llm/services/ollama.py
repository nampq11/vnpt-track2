from typing import Optional, List
from src.brain.llm.services.type import LLMService, LLMServiceConfig
from openai import OpenAI


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
    
    def generate(
        self,
        user_input: str,
        stream: Optional[bool] = False,
    ) -> str:
        """Generate response from Ollama via OpenAI API"""
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """Bạn là một trợ lý thông minh chuyên trả lời câu hỏi trắc nghiệm tiếng Việt.
Hãy đọc kỹ câu hỏi, phân tích các lựa chọn, và chọn đáp án chính xác nhất.
Suy nghĩ theo từng bước 1 để có đáp án chính xác nhất.

Output trả về dưới tag <answer> và không đưa ra giải thích.

Example:
câu hỏi: "Câu hỏi là gì?"
choices: 
- A: Đáp án A
- B: Đáp án B
- C: Đáp án C
- D: Đáp án D

Answer:     
"""},
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
    
    async def embed(self, text: str) -> List[float]:
        """
        Get embedding vector for text using Ollama
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        try:
            response = self.openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding from Ollama: {str(e)}")
    
    def get_all_tools(self):
        """Return available tools"""
        return {}
    
    def get_config(self) -> LLMServiceConfig:
        """Return service configuration"""
        return self.config



