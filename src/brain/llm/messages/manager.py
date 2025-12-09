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

class ContextManager:
    def __init__(
        self,
        formatter: IMessageFormatter
    ) -> None:
        pass