"""Response collection from target LLMs.

Queries multiple LLMs with unsafe prompts and collects their responses
for safety evaluation.
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..llm.base import BaseLLMClient


async def collect_responses_single_model(
    client: BaseLLMClient,
    model_name: str,
    prompts: list[str],
    max_concurrent: int = 5,
) -> list[dict]:
    """Collect responses from a single model for all prompts."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def _query(idx: int, prompt: str) -> dict:
        async with semaphore:
            try:
                response = await client.generate(prompt=prompt, temperature=0.0)
                return {
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "model": model_name,
                    "response": response,
                    "error": "",
                }
            except Exception as e:
                return {
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "model": model_name,
                    "response": "",
                    "error": str(e),
                }

    tasks = [_query(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return list(results)


async def collect_responses_all_models(
    clients: dict[str, BaseLLMClient],
    prompts: list[str],
    max_concurrent_per_model: int = 5,
    save_intermediate: bool = True,
    output_dir: str = "outputs/results",
) -> pd.DataFrame:
    """Collect responses from all target models.

    Args:
        clients: dict mapping model display_name -> client
        prompts: list of unsafe prompts
        max_concurrent_per_model: concurrency limit per model
        save_intermediate: save results per model as they complete
        output_dir: directory for intermediate results

    Returns:
        DataFrame with columns: prompt_idx, prompt, model, response, error
    """
    all_results = []

    for model_name, client in clients.items():
        print(f"\nCollecting responses from {model_name}...")
        results = await collect_responses_single_model(
            client, model_name, prompts, max_concurrent_per_model,
        )
        all_results.extend(results)
        print(f"  Collected {len(results)} responses from {model_name}")

        if save_intermediate:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(output_dir) / f"responses_{model_name.replace(' ', '_').replace('/', '_')}.jsonl"
            with open(save_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(all_results)
    return df
