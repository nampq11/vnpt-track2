import asyncio
import json
from nt import error
import time
from typing import Any, List, Optional
import uuid

from pydantic_core.core_schema import CallSchema
from brain.llm.messages.manager import ContextManager, InternalMessage
from brain.llm.utils.tool_result_formatter import format_tool_result
from src.brain.llm.services.type import LLMService, LLMServiceConfig
from openai import OpenAI
from loguru import logger


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
        context_manager: ContextManager | None = None,
        max_iterations: int = 5,
    ) -> None:
        self.openai = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.context_manager = context_manager
        self.config = OllamaServiceConfig(model=model)
    
    async def generate(
        self,
        user_input: str,
        stream: Optional[bool] = False,
    ) -> str:
        """Generate response from Ollama via OpenAI API"""
        await self.context_manager.add_user_message(
            text_content=user_input,
        )

        message_id = uuid.uuid4()
        start_time = time.time()

        session_id = getattr(self.context_manager, "session_id", None) if hasattr(self, "context_manager") else None

        if (session_id):
            logger.debug(
                "Starting LLM call for session: %s, message_id: %s",
            )

        iteration_count = 0
        try:
            while iteration_count < self.max_iterations:
                iteration_count += 1

                message = await self._get_AI_response_with_retries(
                    tools=[],
                    user_input=user_input,
                )

                if (not message.tool_calls or len(message.tool_calls) == 0):
                    response_text = message.content or ''
                    
                    await self.context_manager.add_assistant_message(
                        content=response_text,
                    )

                    return response_text
                
                if (message.content and message.content.strip()):
                    logger.info(f"thinking: ${message.content.strip()}")

                await self.context_manager.add_assistant_message(
                    message.content,
                    message.tool_calls,
                )

                # handle tool Calls
                for tool_call in message.tool_calls:
                    logger.debug(f"Tool call initiated: ${json.dumps(tool_call, indent=2)}")
                    logger.info(f"Using tool: ${tool_call.function.name}")

                    tool_name = tool_call.function.name
                    args = {}
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception as e:
                        logger.error(f"Error parsing arguments for ${tool_name}:", e)
                        await self.context_manager.add_tool_result(
                            tool_call.id,
                            tool_name,
                            {
                                "error": f"Failed to parse arguments: ${e}",
                            }
                        )
                        continue
                
                    try:
                        result = self.ToolManager.execute_tool(
                            tool_name,
                            args,
                            session_id,
                        )

                        formatted_result = format_tool_result(
                            tool_name,
                            result,
                        )

                        logger.info(f"Tool result: \n${formatted_result}")

                        await self.context_manager.add_tool_result(
                            tool_call.id,
                            tool_name,
                            result,
                        )
                    except Exception as e:
                        logger.error(f"Error executing tool ${tool_name}:", e)
                        await self.context_manager.add_tool_result(
                            tool_call.id,
                            tool_name,
                            {
                                "error": str(e),
                            }
                        )

        except Exception as e:
            logger.error(

            )


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
            raise RuntimeError(f"Error generating response from Ollama: {str(e)}")
    
    def get_all_tools(self):
        """Return available tools"""
        return {}
    
    def get_config(self) -> LLMServiceConfig:
        """Return service configuration"""
        return self.config

    async def _get_AI_response_with_retries(
        self,
        tools: List[Any],
        user_input: str,
    ) -> InternalMessage:
        attempts = 0
        MAX_ATTEMPTS = 3

        while attempts < MAX_ATTEMPTS:
            attempts += 1
            
            try:
                formatted_messages = await self.context_manager.get_formatted_messages({
                    "role": "user",
                    "content": user_input,
                })

                response = await self.openai.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    tools=tools if attempts == 1 else [],
                    tool_choice="auto" if attempts == 1 else "none"
                )

                logger.debug(
                    "OLLAMA CHAT COMPLETION RESPONSE: %s",
                    json.dumps(response, indent=2, default=str)
                )

                message = response.choices[0].message if response.choices else None
                if not message:
                    raise Exception("Received empty message from Ollama API.")
                
                return message
            except Exception as e:
                api_error = error
                logger.debug(
                    f"Error in Ollama API call (Attempt {attempts}/{MAX_ATTEMPTS}): ",
                    f"{str(api_error)}",
                    extra={
                        "status": getattr(api_error, 'status', None),
                        "headers": getattr(api_error, 'headers', None),
                    }
                )

                if getattr(api_error, "status", None) == 400:
                    logger.warning(
                        f"Ollama API error - check if model '{self.model}' is available locally."
                        f"Error details: {getattr(api_error, "error", None)}"
                    )
                
                if attempts >= MAX_ATTEMPTS:
                    logger.error(
                        f"Failed to get response from Ollama after {MAX_ATTEMPTS} attempts."
                    )
                    raise

                await asyncio.sleep(0.5 * attempts)
            
        raise Exception("Failed to get response after maximum attempts.")
