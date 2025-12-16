"""Configuration management for inference pipeline"""
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class OllamaConfig:
    """Ollama service configuration"""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3:1.7b"
    timeout: int = 120
    max_retries: int = 3


@dataclass
class AzureConfig:
    """Azure OpenAI service configuration"""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    azure_endpoint: Optional[str] = None
    model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-ada-002"


@dataclass
class InferenceConfig:
    """Inference pipeline configuration"""
    batch_size: int = 1
    system_prompt: Optional[str] = None
    use_caching: bool = False
    save_intermediate: bool = False
    verbose: bool = True


@dataclass
class DataConfig:
    """Data paths configuration"""
    test_file: str = "data/test.json"
    val_file: str = "data/val.json"
    output_dir: str = "results"
    
    def get_output_file(self, dataset: str = "test") -> str:
        """Get output file path for dataset"""
        return f"{self.output_dir}/predictions_{dataset}.json"


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.ollama = OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
        )
        self.inference = InferenceConfig(
            verbose=os.getenv("VERBOSE", "True").lower() == "true",
        )
        self.data = DataConfig()
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary"""
        config = cls()
        
        if "ollama" in config_dict:
            for key, value in config_dict["ollama"].items():
                setattr(config.ollama, key, value)
        
        if "inference" in config_dict:
            for key, value in config_dict["inference"].items():
                setattr(config.inference, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                setattr(config.data, key, value)
        
        return config

