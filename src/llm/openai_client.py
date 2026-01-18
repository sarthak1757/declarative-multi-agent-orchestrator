import json
import os
from openai import AsyncOpenAI
from .base import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, model="gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _get_tool_schema(self, func):
        # Basic schema generation - in a real app this would be more robust
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "No description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "key": {"type": "string"},
                        "value": {"type": "string"},
                        "expression": {"type": "string"},
                        "code": {"type": "string"},
                    },
                },
            },
        }

    async def generate(self, prompt: str, tools=None) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        tool_schemas = [self._get_tool_schema(t) for t in tools] if tools else None

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tool_schemas,
        )

        message = response.choices[0].message
        
        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                # Find the matching function
                tool_func = next((t for t in tools if t.__name__ == func_name), None)
                if tool_func:
                    result = tool_func(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    })
            
            # Get final response
            final_response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return final_response.choices[0].message.content

        return message.content