from src.brain.system_prompt.interface import PromptGenerationResult, ProviderResult

class EnhancedPromptManager:
    def __init__(self) -> None:
        pass

    async def generate_system_prompt(self) -> PromptGenerationResult:
        return PromptGenerationResult(
            content="",
            provider_results=[ProviderResult(
                provider_id="",
                content="",
                generation_time_ms=0,
                success=True,
                error=None
            )],
            generation_time_ms=0,
            success=True
        )
