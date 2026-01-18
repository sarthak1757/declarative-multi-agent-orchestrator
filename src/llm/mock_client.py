from .base import BaseLLMClient
from src.tools.memory_tool import memory_tool

class MockClient(BaseLLMClient):
    def __init__(self, model="mock-model"):
        self.model = model

    async def generate(self, prompt: str, tools=None) -> str:
        if "memory" in prompt.lower() or "analyze" in prompt.lower():
            # Simulate tool usage
            memory_tool("write", "test_key", "helo")
            return "Mock response: I have written to memory."
        return f"Mock response for: {prompt[:50]}..."