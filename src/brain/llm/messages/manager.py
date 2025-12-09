from ast import List
from dataclasses import dataclass
from typing import Any, Optional, Protocol


class IMessageFormatter(Protocol):
    def format(
        self,
        message: Any,
        _: Any
    ) -> Any:
        ...
    def parse_response(
        self,
        response: Any,
    ) -> Optional[Any]:
        ...
    
    def parse_stream_response(
        self,
        response: Any,
    ) -> Optional[Any]:
        ...

class IConversationHistoryProvider(Protocol):
    def get_history(
        self,
        session_id: str
    ) -> List[Any]:
        ...
    async def save_message(
        self,
        session_id: str,
        message: Any,
    ) -> None:
        ...

@dataclass
class InternalMessage:
    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ContextManager:
    def __init__(
        self,
        formatter: IMessageFormatter,
        prompt_manager: Any,
        history_provider: Optional[IConversationHistoryProvider],
        session_id: Optional[str] = None,
    ):
        if not formatter:
            raise ValueError("Formatter is required")

        self.formatter = formatter
        self.prompt_manager = prompt_manager
        self.history_provider = history_provider
        self.session_id = session_id

        self.messages: List[InternalMessage] = []

        self.current_token_count = 0

    async def get_system_prompt(
        self,
    ) -> str:
        result = await self.prompt_manager.generate_system_prompt()
        return result.content