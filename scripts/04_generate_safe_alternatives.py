#!/usr/bin/env python3
"""Script 04: Generate safe alternatives for unsafe prompts (Box B2 template).

Creates safe versions of unsafe prompts by removing risk-triggering elements
while preserving topic and structure.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset
from src.generation.safe_alternative import generate_all_safe_alternatives
from src.llm.registry import get_client


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load filtered prompts
    input_path = Path(results_dir) / "filtered_prompts.jsonl"
    if input_path.exists():
        prompts = []
        with open(input_path) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    prompts.append(obj.get("candidate_prompt", obj.get("prompt", "")))
    else:
        # Fall back to dataset
        df = load_dataset(config.dataset_path)
        prompts = df["unsafe_prompts"].tolist()[:args.limit]

    print(f"Generating safe alternatives for {len(prompts)} prompts...")

    # Create client
    api_key = config.api_keys.get("openai", "")
    client = get_client("gpt-5.1", "openai", api_key)

    # Generate safe alternatives
    safe_prompts = await generate_all_safe_alternatives(
        client, prompts, max_concurrent=args.concurrent,
    )

    # Save results
    output_path = Path(results_dir) / "safe_alternatives.csv"
    df_out = pd.DataFrame({
        "unsafe_prompt": prompts,
        "safe_prompt": safe_prompts,
    })
    df_out.to_csv(output_path, index=False)

    print(f"Generated {len(safe_prompts)} safe alternatives")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate safe alternatives")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
