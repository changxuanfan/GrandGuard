"""Google Generative AI client.

Supports: Gemini-2.5-Flash.
"""

import google.generativeai as genai

from .base import BaseLLMClient


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(model_name, api_key, base_url)
        genai.configure(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        async def _call():
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt if system_prompt else None,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            response = await model.generate_content_async(prompt)
            return response.text if response.text else ""

        return await self._retry_with_backoff(_call)
