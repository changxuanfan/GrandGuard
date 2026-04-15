"""xAI LLM client (OpenAI-compatible).

Supports: Grok-4, Grok-4.1.
"""

from .openai_client import OpenAIClient


class XAIClient(OpenAIClient):
    """Client for xAI Grok models via OpenAI-compatible API."""

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url or self.XAI_BASE_URL,
        )
