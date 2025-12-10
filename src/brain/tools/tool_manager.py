
from ast import Dict
from dataclasses import dataclass
from loguru import logger
from typing import Any, Optional, TypedDict


@dataclass
class ToolManagerConfig:
    excution_timeout: Optional[int] = None

class ToolSpec(TypedDict):
    description: str
    parameters: Any

CombinedToolSet = Dict[str, ToolSpec]

class ToolManager:
    def __init__(
        self,
        config: ToolManagerConfig = {},
    ):
        self.config = ToolManagerConfig(
            excution_timeout=30000,
            **config,
        )
        self.initialized = False
    
    async def initialize(
        self,
    ):
        if self.initialized:
            logger.warning("Internal tool Manager: Already initialized")
            return
        

    async def get_all_tools(
        self,
    ) -> CombinedToolSet:
        combined_tools: CombinedToolSet = {}
        logger.debug("Tool Manager: Tools config args", self.config)

        return combined_tools

    async def execute_tool(
        tool_name: str,
        args: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        pass

    def _ensure_initialized(self):
        if (not self.initialized):
            raise Exception("Tool Manager not initialized")