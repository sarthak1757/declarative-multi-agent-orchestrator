import os
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable
from .base import BaseLLMClient

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class GeminiClient(BaseLLMClient):
    def __init__(self, model="models/gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model)

    async def generate(self, prompt: str, tools=None) -> str:
        # Gemini SDK handles function schema generation if we pass the functions directly
        chat = self.model.start_chat(enable_automatic_function_calling=True)
        
        # If tools are provided, we need to configure the chat with them
        if tools:
            # Re-initialize chat with tools
            # Note: enable_automatic_function_calling=True handles the loop automatically in the Python SDK!
            chat = self.model.start_chat(
                tools=tools,
                enable_automatic_function_calling=True
            )
            
        response = chat.send_message(prompt)
        return response.text