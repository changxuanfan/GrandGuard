"""OpenAI-compatible LLM client.

Supports: GPT-5.1, GPT-4.1-Mini, GPT-OSS-120B, GPT-OSS-Safeguard-20B,
          Llama-4-Maverick (via OpenAI-compatible endpoint).
"""

from openai import AsyncOpenAI

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible APIs."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(model_name, api_key, base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async def _call():
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await self._retry_with_backoff(_call)
