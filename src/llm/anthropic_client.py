"""Anthropic LLM client.

Supports: Claude-Sonnet-4.5, Claude-Sonnet-3.7.
"""

from anthropic import AsyncAnthropic

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(model_name, api_key, base_url)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature > 0:
            kwargs["temperature"] = temperature

        async def _call():
            response = await self.client.messages.create(**kwargs)
            return response.content[0].text if response.content else ""

        return await self._retry_with_backoff(_call)
