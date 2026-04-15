#!/usr/bin/env python3
"""Script 02: Generate unsafe prompts using LLM (Box B3 template).

Uses Grok-4 to generate candidate unsafe prompts for each third-level
risk type using few-shot prompting with seed prompts from the dataset.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset
from src.generation.prompt_generator import generate_all_risk_types
from src.llm.registry import get_client
from src.taxonomy import SECOND_LEVEL


def extract_seed_prompts(df, n_seeds_per_type: int = 10) -> dict[str, list[str]]:
    """Extract seed prompts from the dataset grouped by risk type."""
    seeds = {}
    for rt in df["risk_type"].unique():
        rt_prompts = df[df["risk_type"] == rt]["unsafe_prompts"].tolist()
        seeds[rt] = rt_prompts[:n_seeds_per_type]
    return seeds


def compute_target_counts(df, total_target: int = 10000) -> dict[str, int]:
    """Compute target prompt counts per risk type, proportional to dataset distribution."""
    dist = df["risk_type"].value_counts()
    total = dist.sum()
    counts = {}
    for rt, cnt in dist.items():
        counts[rt] = max(10, round(total_target * cnt / total))
    return counts


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset for seed prompts
    df = load_dataset(config.dataset_path)
    seed_data = extract_seed_prompts(df)

    # Create generator client (Grok-4)
    api_key = config.api_keys.get("xai", "")
    client = get_client(config.generator_model, "xai", api_key)

    # Compute target counts
    target_counts = compute_target_counts(df, args.total)
    print(f"Generating {sum(target_counts.values())} prompts across {len(target_counts)} risk types...")

    # Generate
    all_prompts = await generate_all_risk_types(client, seed_data, target_counts)

    # Save
    output_path = Path(results_dir) / "raw_candidates.jsonl"
    with open(output_path, "w") as f:
        for p in all_prompts:
            f.write(json.dumps(p) + "\n")

    print(f"\nGenerated {len(all_prompts)} candidate prompts")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate unsafe prompts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--total", type=int, default=10000, help="Total prompts to generate")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
