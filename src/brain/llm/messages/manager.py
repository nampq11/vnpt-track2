from ast import List
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from openai.types.vector_store_search_response import Content

from src.brain.system_prompt.enhanced_manager import EnhancedPromptManager
from src.brain.system_prompt.interface import PromptGenerationResult


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
        prompt_manager: EnhancedPromptManager,
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
    
    async def add_message(
        self,
        message: InternalMessage
    ):
        self._validate_message(message)
        await self.store_message(message)

    async def add_user_message(
        self,
        text_content: str
    ):
        if not isinstance(text_content, str) or text_content.strip() == '':
            raise ValueError('Content must be non-empty string.')
        
        message_parts = self._build_user_message_content(
            text_content
        )

        await self.add_message(InternalMessage(
            role='user',
            content=message_parts
        ))

    async def add_assistant_message(
        self,
        content: str | None,
        tool_calls: InternalMessage.tool_calls,
    ):
        if content == None and (not tool_calls or len(tool_calls) == 0):
            raise ValueError('Must provide content or tool calls')
        
        await self.add_message(
            InternalMessage(
                role='assistant',
                content=content,
                tool_calls=tool_calls if (tool_calls and len(tool_calls) > 0) else None,
            )
        )

    async def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        result: Any
    ):
        if not tool_call_id or not name:
            raise ValueError('add_tool_result: tool_call_id and name are required.')
        
        content = self.format_tool_result_content(result);
        await self.add_message(
            InternalMessage(
                role='tool',
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            )
        )

    async def get_formatted_messages(
        self,
        _message: InternalMessage
    ):
        try:
            return self.get_all_formatted_messages()
        except Exception as err:
            raise ValueError(f'Failed to format message: ${err}')

    async def get_all_formatted_messages(
        self,
        include_system_message: bool = True,
    ):
        try:
            formatted_messages = []
            prompt = await self.get_system_prompt()
            if include_system_message and prompt:
                formatted_messages.append(
                    InternalMessage(
                        role='system',
                        content=prompt,
                    )
                )
            
            validated_messages = self.validate_and_repair_message_flow(
                self.messages
            )

            for msg in validated_messages:
                formatted = self.formatter.format(msg, None)

                if isinstance(formatted, list):
                    formatted_messages.extend(formatted)
                elif formatted and formatted != None:
                    formatted_messages.append(formatted)
            
            return formatted_messages
        except Exception as err:
            raise ValueError(f'Failed to get all formatted messages: ${err}')

    def _validate_message(
        self,
        message: InternalMessage
    ): 
        if not message.role:
            raise ValueError('Role is required for a message')
        
        match message.role:
            case 'user':
                self._validate_user_message(message)
            case 'assistant':
                self._validate_assistant_message(message)
            case 'tool':
                self._validate_tool_message(message)
            case _:
                raise ValueError(f'Invalid role: {message.role}')

    def _validate_user_message(
        self,
        message: InternalMessage
    ):
        has_valid_array_content = isinstance(message.content, list) and len(message.content) > 0
        has_valid_string_content = (
            isinstance(message.content, str) and message.content.strip() != ""
        )

        if not has_valid_array_content and not has_valid_string_content:
            raise ValueError(
                'User message content should be a non-empty string or a non-empty array of parts.'
            )
    
    def _validate_assistant_message(
        self,
        message: InternalMessage
    ):
        if (message.content == None and (not message.tool_calls or len(message.tool_calls) == 0)):
            raise ValueError('Assistant message content must have content or tool calls.')
        
        if message.tool_calls and not self._is_valid_tool_calls(message.tool_calls):
            raise ValueError('Invalid tool calls structure in assistant message.')

    def _validate_tool_message(
        self,
        message: InternalMessage
    ):
        if not message.tool_call_id or not message.name or message.content == None:
            raise ValueError('Tool message missing required fields (tool_call_id, name, content).')

    def _is_valid_tool_calls(
        self,
        tool_calls: List[Any]
    ) -> bool:
        return (
            isinstance(tool_calls, list) and not any(
                not tc.get('id') or not tc.get('function', {}).get('name') or
                not tc.get('function', {}).get('arguments')
            )
            for tc in tool_calls
        )
    
    async def _store_message(
        self,
        message: InternalMessage
    ): 
        if (self._should_use_persistent_storage()):
            try:
                await self.history_provider.save_message(
                    self.session_id,
                    message
                )
                self.messages.append(message)
            except Exception:
                self.messages.append(message)
        else:
            self.messages.append(message)
    
    def _should_use_persistent_storage(self) -> bool:
        has_history_provider = bool(self.history_provider)
        has_session_id = bool(self.session_id)
        
        should_use = has_history_provider and has_session_id
        return should_use

    def _validate_and_repair_message_flow(
        self,
        messages: List[InternalMessage]
    ) -> List[InternalMessage]:
        repaired_messages: List[InternalMessage] = []
        last_assistant_with_tool_calls: InternalMessage | None = None
        orphaned_tool_messages = 0

        for (index, message) in enumerate[InternalMessage](messages):
            if not message:
                continue
            
            match message.role:
                case 'system':
                    pass
                case 'user':
                    repaired_messages.append(message)
                    last_assistant_with_tool_calls = None
                    break
                case 'assistant':
                    repaired_messages.append(message)
                    if message.tool_calls and len(message.tool_calls) > 0:
                        last_assistant_with_tool_calls = message
                    else:
                        last_assistant_with_tool_calls = None
                    break
                case 'tool':
                    if last_assistant_with_tool_calls and self._is_valid_tool_response(message, last_assistant_with_tool_calls):
                        repaired_messages.append(message)
                    else:
                        orphaned_tool_messages+=1
                    break
                case _:
                    repaired_messages.append(message)
                    break

    def _is_valid_tool_response(
        self,
        tool_message: InternalMessage,
        assistant_message: InternalMessage,
    ) -> bool:
        if not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0:
            return False

        matching_tool_call = next(
            (tc for tc in assistant_message.tool_calls if tc.id == tool_message.tool_call_id),
            None
        )

        return matching_tool_call is not None

