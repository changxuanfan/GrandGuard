#!/usr/bin/env python3
"""Script 03: Filter generated prompts using LLM-judge (Box B4 template).

Applies the GPT-5.1 judge filter to retain only high-quality unsafe prompts.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.generation.judge_filter import filter_batch
from src.llm.registry import get_client


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load raw candidates
    input_path = Path(results_dir) / "raw_candidates.jsonl"
    candidates = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    print(f"Loaded {len(candidates)} raw candidates")

    # Create judge client (GPT-5.1)
    api_key = config.api_keys.get("openai", "")
    client = get_client(config.judge_filter_model, "openai", api_key)

    # Filter in batches
    batch_size = 100
    all_kept = []
    all_discarded = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        kept, discarded = await filter_batch(client, batch, max_concurrent=args.concurrent)
        all_kept.extend(kept)
        all_discarded.extend(discarded)
        print(f"  Batch {i // batch_size + 1}: kept {len(kept)}/{len(batch)}")

    # Save filtered prompts
    output_path = Path(results_dir) / "filtered_prompts.jsonl"
    with open(output_path, "w") as f:
        for p in all_kept:
            f.write(json.dumps(p) + "\n")

    # Save discard log
    discard_path = Path(results_dir) / "discarded_prompts.jsonl"
    with open(discard_path, "w") as f:
        for p in all_discarded:
            f.write(json.dumps(p) + "\n")

    print(f"\nKept {len(all_kept)}/{len(candidates)} prompts ({len(all_kept)/len(candidates)*100:.1f}%)")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Filter generated prompts")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
