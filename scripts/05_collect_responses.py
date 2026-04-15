#!/usr/bin/env python3
"""Script 05: Collect responses from 10 target LLMs.

Samples 500 unsafe prompts and queries all target LLMs to collect responses.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset, sample_unsafe_prompts
from src.generation.response_collector import collect_responses_all_models
from src.llm.registry import get_all_target_clients


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load and sample prompts
    df = load_dataset(config.dataset_path)
    sampled = sample_unsafe_prompts(df, n=args.num_prompts, seed=config.seed)
    prompts = sampled["unsafe_prompts"].tolist()

    print(f"Sampled {len(prompts)} unsafe prompts for response collection")

    # Create target LLM clients
    clients = get_all_target_clients(config)
    print(f"Target models: {list(clients.keys())}")

    # Collect responses
    results_df = await collect_responses_all_models(
        clients, prompts,
        max_concurrent_per_model=args.concurrent,
        save_intermediate=True,
        output_dir=results_dir,
    )

    # Save combined results
    output_path = Path(results_dir) / "responses.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nCollected {len(results_df)} total responses")
    print(f"Models: {results_df['model'].nunique()}")
    print(f"Errors: {(results_df['error'] != '').sum()}")
    print(f"Saved to {output_path}")

    # Save the sampled prompts for reference
    sampled.to_csv(Path(results_dir) / "sampled_prompts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Collect LLM responses")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
