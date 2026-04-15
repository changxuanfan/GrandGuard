"""DeepSeek LLM client (OpenAI-compatible).

Supports: DeepSeek-V3.2.
"""

from .openai_client import OpenAIClient


class DeepSeekClient(OpenAIClient):
    """Client for DeepSeek models via OpenAI-compatible API."""

    DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url or self.DEEPSEEK_BASE_URL,
        )
