"""Enhanced Prompt Manager for loading and managing system prompts."""
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.brain.system_prompt.interface import PromptGenerationResult, ProviderResult
from src.brain.system_prompt.registry import PromptRegistry, PromptType


class EnhancedPromptManager:
    """Manager for loading prompts from markdown files and populating registry."""
    
    _instance: Optional["EnhancedPromptManager"] = None
    
    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self.prompts_dir = prompts_dir or Path(__file__).parent / "files"
        self.registry = PromptRegistry.get_instance()
        self._loaded = False
        self._base_system_prompt: str = ""
    
    @classmethod
    def get_instance(cls) -> "EnhancedPromptManager":
        """Get singleton instance of EnhancedPromptManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
        PromptRegistry.reset_instance()
    
    def ensure_loaded(self) -> None:
        """Ensure all prompts are loaded."""
        if not self._loaded:
            self._load_all()
    
    def _load_all(self) -> None:
        """Load all prompt files from the prompts directory."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Load base system prompt if exists
        system_file = self.prompts_dir / "system.md"
        if system_file.exists():
            self._base_system_prompt = system_file.read_text(encoding="utf-8").strip()
        
        # Load all prompt files
        for file_path in self.prompts_dir.glob("*.md"):
            if file_path.name == "system.md":
                continue  # Skip base system prompt
            self._load_prompt_file(file_path)
        
        self._loaded = True
    
    def _load_prompt_file(self, file_path: Path) -> None:
        """Load and parse a single prompt markdown file."""
        content = file_path.read_text(encoding="utf-8")
        prompts = self._parse_prompt_sections(content)
        
        for prompt_type, (system, user) in prompts.items():
            self.registry.register(prompt_type, system, user)
    
    def _parse_prompt_sections(self, content: str) -> Dict[str, Tuple[str, str]]:
        """
        Parse markdown content into prompt sections.
        
        Format:
        ---
        type: prompt_type_name
        ---
        
        ## SYSTEM
        system prompt content...
        
        ## USER
        user prompt template...
        """
        prompts: Dict[str, Tuple[str, str]] = {}
        
        # Split by prompt type blocks (--- type: xxx ---)
        type_pattern = r'---\s*\ntype:\s*(\w+)\s*\n---'
        blocks = re.split(type_pattern, content)
        
        # blocks[0] is content before first type marker (usually empty)
        # blocks[1], blocks[2] are type name, content pairs
        i = 1
        while i < len(blocks) - 1:
            prompt_type = blocks[i].strip()
            block_content = blocks[i + 1]
            
            system_prompt, user_prompt = self._extract_system_user(block_content)
            if system_prompt or user_prompt:
                prompts[prompt_type] = (system_prompt, user_prompt)
            
            i += 2
        
        return prompts
    
    def _extract_system_user(self, content: str) -> Tuple[str, str]:
        """Extract SYSTEM and USER sections from a block."""
        system_prompt = ""
        user_prompt = ""
        
        # Find ## SYSTEM section (ends at ## USER or end of content)
        system_match = re.search(
            r'##\s*SYSTEM\s*\n(.*?)(?=##\s*USER|$)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if system_match:
            system_prompt = system_match.group(1).strip()
        
        # Find ## USER section (ends at next --- type block or end of content)
        # Use greedy match since we want everything until the next block or EOF
        user_match = re.search(
            r'##\s*USER\s*\n(.+?)(?=\n---\s*\ntype:|$)',
            content,
            re.DOTALL | re.IGNORECASE
        )
        if user_match:
            user_prompt = user_match.group(1).strip()
        else:
            # Fallback: try to get everything after ## USER until end
            user_fallback = re.search(
                r'##\s*USER\s*\n(.+)',
                content,
                re.DOTALL | re.IGNORECASE
            )
            if user_fallback:
                user_prompt = user_fallback.group(1).strip()
        
        return system_prompt, user_prompt
    
    def get_prompt(self, prompt_type: PromptType) -> Tuple[str, str]:
        """Get system and user prompts for a given type."""
        self.ensure_loaded()
        return self.registry.get_prompts(prompt_type)
    
    def get_system_prompt(self, prompt_type: PromptType) -> str:
        """Get system prompt for a given type."""
        self.ensure_loaded()
        return self.registry.get_system(prompt_type)
    
    def get_user_prompt(self, prompt_type: PromptType) -> str:
        """Get user prompt template for a given type."""
        self.ensure_loaded()
        return self.registry.get_user(prompt_type)

    async def generate_system_prompt(self) -> PromptGenerationResult:
        """
        Generate base system prompt for ContextManager compatibility.
        Returns the content from system.md file.
        """
        start_time = time.perf_counter()
        
        try:
            self.ensure_loaded()
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            
            return PromptGenerationResult(
                content=self._base_system_prompt,
                provider_results=[
                    ProviderResult(
                        provider_id="file",
                        content=self._base_system_prompt,
                        generation_time_ms=elapsed_ms,
                success=True,
                error=None
                    )
                ],
                generation_time_ms=elapsed_ms,
                success=True,
                errors=[]
        )
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            return PromptGenerationResult(
                content="",
                provider_results=[
                    ProviderResult(
                        provider_id="file",
                        content="",
                        generation_time_ms=elapsed_ms,
                        success=False,
                        error=e
                    )
                ],
                generation_time_ms=elapsed_ms,
                success=False,
                errors=[e]
            )
