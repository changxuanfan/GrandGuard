"""Qwen LLM client (OpenAI-compatible via DashScope).

Supports: Qwen3-Max.
"""

from .openai_client import OpenAIClient


class QwenClient(OpenAIClient):
    """Client for Qwen models via DashScope OpenAI-compatible API."""

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url or self.DASHSCOPE_BASE_URL,
        )
