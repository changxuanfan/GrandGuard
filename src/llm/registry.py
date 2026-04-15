"""LLM client registry: maps (model_name, provider) to client instances."""

from ..config import Config, ModelSpec
from .anthropic_client import AnthropicClient
from .base import BaseLLMClient
from .deepseek_client import DeepSeekClient
from .google_client import GoogleClient
from .openai_client import OpenAIClient
from .qwen_client import QwenClient
from .xai_client import XAIClient

PROVIDER_CLIENT_MAP = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "google": GoogleClient,
    "deepseek": DeepSeekClient,
    "qwen": QwenClient,
    "xai": XAIClient,
}

# Map provider names to API key config names
PROVIDER_KEY_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "xai": "xai",
}


def get_client(model_name: str, provider: str, api_key: str) -> BaseLLMClient:
    """Create an LLM client instance for the given model and provider."""
    client_cls = PROVIDER_CLIENT_MAP.get(provider)
    if client_cls is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDER_CLIENT_MAP.keys())}")
    return client_cls(model_name=model_name, api_key=api_key)


def get_client_from_config(model_spec: ModelSpec, config: Config) -> BaseLLMClient:
    """Create an LLM client from a ModelSpec and Config."""
    key_name = PROVIDER_KEY_MAP.get(model_spec.provider, model_spec.provider)
    api_key = config.api_keys.get(key_name, "")
    return get_client(model_spec.name, model_spec.provider, api_key)


def get_all_target_clients(config: Config) -> dict[str, BaseLLMClient]:
    """Create client instances for all target LLMs defined in config.

    Returns a dict mapping display_name -> client.
    """
    clients = {}
    for spec in config.target_llms:
        clients[spec.display_name] = get_client_from_config(spec, config)
    return clients
