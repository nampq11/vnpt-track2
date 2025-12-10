
from dataclasses import dataclass
import json
from typing import Any, Optional
from loguru import logger

@dataclass
class GenericToolResult:
    success: Optional[bool] = None
    message: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Any] = None
    content: Optional[Any] = None


def fmt_color(color: str, text: str) -> str:
    colors = {
        "gray": "<gray>",
        "red": "<red>",
        "green": "<green>",
        "yellow": "<yellow>",
        "blue": "<blue>",
        "reset": "</reset>",
    }
    return f"{colors.get(color, "")}{text}{colors["reset"]}"

def format_tool_result(
    tool_name: str,
    result: Any
) -> str:
    if not result:
        return fmt_color("gray", "(No result)")
    
    if isinstance(result, str):
        return result

    try:
        return format_generic_result(result)
    except Exception as e:
        logger.debug(
            "Error formatting tool result, falling back to JSON",
            extra={
                "error": str(e),
                "tool_name": tool_name,
            }
        )

        if isinstance(result, dict) and result.get("error"):
            return fmt_color("red", f"Error: {result["error"]}")
        
        return json.dumps(result, indent=2, ensure_ascii=False)

def format_generic_result(
    result: GenericToolResult
) -> str:
    if result.success == False or result.error:
        return fmt_color("red", f"Error: ${result.error or result.message or 'Operation failed'}")
    
    has_data = result.data or (result.content and not result.error)
    is_explicit_success = result.success == True
    
    if is_explicit_success or has_data:
        output = []

        if result.data:
            data = result["data"]

            if isinstance(data, dict):
                keys = list(data.keys())

                if len(keys) > 0:
                    output.append("")
                    output.append(fmt_color("gray", "Data:"))

                    for key in keys[:5]:
                        value = data[key]

                        if isinstance(value, str):
                            display_value = value
                        else:
                            display_value = json.dumps(value, ensure_ascii=False)
                        
                        truncated = display_value[:50] + "..." if len(display_value) > 50 else ""
                        output.append(fmt_color("gray", f"  {key}: {truncated}"))

                    if len(keys) > 5:
                        output.append(fmt_color("gray", f"... and {len(keys -5)} more fields"))

        return "\n".join(output)
    
    return ""

