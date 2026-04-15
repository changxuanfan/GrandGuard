"""Abstract base class for LLM clients."""

import asyncio
import time
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for all LLM API clients."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a single completion."""
        ...

    def generate_sync(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Synchronous wrapper for generate."""
        return asyncio.run(self.generate(prompt, system_prompt, temperature, max_tokens))

    async def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Generate completions for a batch of prompts with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_generate(p: str) -> str:
            async with semaphore:
                return await self.generate(p, system_prompt, temperature, max_tokens)

        tasks = [limited_generate(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def generate_batch_sync(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Synchronous wrapper for generate_batch."""
        return asyncio.run(
            self.generate_batch(prompts, system_prompt, temperature, max_tokens, max_concurrent)
        )

    async def _retry_with_backoff(self, coro_func, max_retries: int = 5, base_delay: float = 1.0):
        """Retry an async coroutine with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await coro_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                print(f"[{self.model_name}] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
