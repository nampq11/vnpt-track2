"""Unified configuration for Titan Shield RAG System"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
from pathlib import Path


@dataclass
class OllamaConfig:
    """Ollama service configuration (for development)"""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3:1.7b"
    timeout: int = 120
    max_retries: int = 3


@dataclass
class VNPTModelCredentials:
    """Credentials for a specific VNPT model"""
    api_key: str                           # Bearer token from VNPT portal
    token_id: str                          # Token-id from VNPT portal
    token_key: str                         # Token-key from VNPT portal
    llm_api_name: Optional[str] = None     # Model identifier (LLM small, LLM large, etc)


@dataclass
class VNPTConfig:
    """VNPT API configuration (for production Docker)
    
    Per VNPT API Documentation:
    - Chat endpoints: /data-service/v1/chat/completions/vnptai-hackathon-{small|large}
    - Embedding endpoint: /data-service/vnptai-hackathon-embedding
    - Base URL: https://api.idg.vnpt.vn
    - Auth: Bearer token (api_key) + Token-id + Token-key headers
    
    Supports per-model credentials:
    - credentials_embedding: For embedding model
    - credentials_small: For small chat model
    - credentials_large: For large chat model
    """
    credentials_embedding: Optional[VNPTModelCredentials] = None
    credentials_small: Optional[VNPTModelCredentials] = None
    credentials_large: Optional[VNPTModelCredentials] = None
    request_timeout: int = 30              # Timeout in seconds
    model_size: str = "small"              # Default model size: "small" or "large"
    
    def get_credentials(self, model_type: str = "small") -> Optional[VNPTModelCredentials]:
        """Get credentials for a specific model type
        
        Args:
            model_type: One of "small", "large", "embedding"
            
        Returns:
            VNPTModelCredentials or None if not configured
        """
        if model_type == "embedding":
            return self.credentials_embedding
        elif model_type == "small":
            return self.credentials_small
        elif model_type == "large":
            return self.credentials_large
        return None


@dataclass
class AzureConfig:
    """Azure OpenAI configuration (for offline enrichment)"""
    api_key: Optional[str] = None
    api_version: str = "2024-02-15-preview"
    endpoint: Optional[str] = None
    model: str = "gpt-4"


@dataclass
class OfflineConfig:
    """Configuration for offline data pipeline"""
    output_dir: str = "src/artifacts"
    batch_size: int = 32
    chunk_size: int = 512
    chunk_overlap: int = 128
    enable_enrichment: bool = True
    crawl_depth: int = 2


@dataclass
class RuntimeConfig:
    """Configuration for runtime inference engine"""
    artifacts_dir: str = "src/artifacts"
    safety_threshold: float = 0.85
    max_chunks_retrieved: int = 5
    min_relevance_score: float = 0.3
    bm25_weight: float = 0.6
    vector_weight: float = 0.4


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
    """Main unified configuration class"""
    
    def __init__(self):
        """Initialize configuration from environment variables"""
        self.llm_provider = os.getenv("LLM_PROVIDER", "ollama")  # ollama, azure, vnpt
        
        # LLM Services
        self.ollama = OllamaConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
        )
        
        # Load VNPT credentials - support both env vars and JSON config file
        self.vnpt = self._load_vnpt_config()
        
        self.azure = AzureConfig(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        
        # Offline & Runtime
        self.offline = OfflineConfig(
            output_dir=os.getenv("ARTIFACTS_DIR", "src/artifacts"),
            enable_enrichment=os.getenv("ENABLE_ENRICHMENT", "true").lower() == "true",
        )
        
        self.runtime = RuntimeConfig(
            artifacts_dir=os.getenv("ARTIFACTS_DIR", "src/artifacts"),
            safety_threshold=float(os.getenv("SAFETY_THRESHOLD", "0.85")),
        )
        
        # Data
        self.data = DataConfig()
        
        # Create artifacts directory if needed
        Path(self.offline.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_vnpt_config(self) -> VNPTConfig:
        """Load VNPT configuration from env variables or JSON config file
        
        Priority:
        1. If VNPT_CONFIG_FILE env var is set, load from JSON file
        2. Otherwise, use individual env variables (backward compatible)
        
        JSON config file format:
        {
            "credentials": [
                {
                    "authorization": "Bearer token...",
                    "tokenId": "uuid",
                    "tokenKey": "key",
                    "llmApiName": "LLM embeddings"
                },
                ...
            ]
        }
        """
        import json
        
        config_file = os.getenv("VNPT_CONFIG_FILE")
        if config_file and Path(config_file).exists():
            # Load from JSON config file
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            
            credentials_dict = {}
            if "credentials" in config_data:
                for cred in config_data["credentials"]:
                    # Extract model type from llmApiName
                    api_name = cred.get("llmApiName", "").lower()
                    
                    model_cred = VNPTModelCredentials(
                        api_key=cred["authorization"].replace("Bearer ", ""),
                        token_id=cred["tokenId"],
                        token_key=cred["tokenKey"],
                        llm_api_name=cred.get("llmApiName"),
                    )
                    
                    # Map by model type
                    if "embedding" in api_name:
                        credentials_dict["embedding"] = model_cred
                    elif "small" in api_name:
                        credentials_dict["small"] = model_cred
                    elif "large" in api_name:
                        credentials_dict["large"] = model_cred
            
            return VNPTConfig(
                credentials_embedding=credentials_dict.get("embedding"),
                credentials_small=credentials_dict.get("small"),
                credentials_large=credentials_dict.get("large"),
                model_size=os.getenv("VNPT_MODEL_SIZE", "small"),
            )
        else:
            # Fallback to individual env variables (backward compatible)
            return VNPTConfig(
                credentials_embedding=VNPTModelCredentials(
                    api_key=os.getenv("VNPT_API_KEY_EMBEDDING", os.getenv("VNPT_API_KEY", "")),
                    token_id=os.getenv("VNPT_TOKEN_ID_EMBEDDING", os.getenv("VNPT_TOKEN_ID", "")),
                    token_key=os.getenv("VNPT_TOKEN_KEY_EMBEDDING", os.getenv("VNPT_TOKEN_KEY", "")),
                ) if os.getenv("VNPT_API_KEY") or os.getenv("VNPT_API_KEY_EMBEDDING") else None,
                credentials_small=VNPTModelCredentials(
                    api_key=os.getenv("VNPT_API_KEY_SMALL", os.getenv("VNPT_API_KEY", "")),
                    token_id=os.getenv("VNPT_TOKEN_ID_SMALL", os.getenv("VNPT_TOKEN_ID", "")),
                    token_key=os.getenv("VNPT_TOKEN_KEY_SMALL", os.getenv("VNPT_TOKEN_KEY", "")),
                ) if os.getenv("VNPT_API_KEY") or os.getenv("VNPT_API_KEY_SMALL") else None,
                credentials_large=VNPTModelCredentials(
                    api_key=os.getenv("VNPT_API_KEY_LARGE", os.getenv("VNPT_API_KEY", "")),
                    token_id=os.getenv("VNPT_TOKEN_ID_LARGE", os.getenv("VNPT_TOKEN_ID", "")),
                    token_key=os.getenv("VNPT_TOKEN_KEY_LARGE", os.getenv("VNPT_TOKEN_KEY", "")),
                ) if os.getenv("VNPT_API_KEY") or os.getenv("VNPT_API_KEY_LARGE") else None,
                model_size=os.getenv("VNPT_MODEL_SIZE", "small"),
            )
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary"""
        config = cls()
        
        # Update fields from dict if provided
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        vnpt_dict = {
            "request_timeout": self.vnpt.request_timeout,
            "model_size": self.vnpt.model_size,
        }
        if self.vnpt.credentials_embedding:
            vnpt_dict["credentials_embedding"] = self.vnpt.credentials_embedding.__dict__
        if self.vnpt.credentials_small:
            vnpt_dict["credentials_small"] = self.vnpt.credentials_small.__dict__
        if self.vnpt.credentials_large:
            vnpt_dict["credentials_large"] = self.vnpt.credentials_large.__dict__
        
        return {
            "llm_provider": self.llm_provider,
            "ollama": self.ollama.__dict__,
            "vnpt": vnpt_dict,
            "azure": self.azure.__dict__,
            "offline": self.offline.__dict__,
            "runtime": self.runtime.__dict__,
            "data": self.data.__dict__,
        }

