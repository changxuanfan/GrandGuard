#!/usr/bin/env python3
"""Script 01: Analyze the GrandGuard benchmark dataset.

Loads the dataset, computes statistics (row counts, risk type distribution),
and saves results.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset, get_statistics


def main():
    parser = argparse.ArgumentParser(description="Analyze GrandGuard dataset")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GrandGuard Dataset Analysis")
    print("=" * 60)

    # Load dataset
    df = load_dataset(config.dataset_path)
    stats = get_statistics(df)

    print(f"\nTotal rows: {stats['total_rows']}")
    print(f"Total unsafe prompts: {stats['total_unsafe_prompts']}")
    print(f"Total safe prompts: {stats['total_safe_prompts']}")
    print(f"Rows with responses: {stats['rows_with_responses']}")

    print("\n--- Risk Type Distribution ---")
    for code in sorted(stats["second_level_distribution"].keys()):
        info = stats["second_level_distribution"][code]
        print(f"  {code} ({info['name']}): {info['count']} prompts")

    # Compute benchmark composition (10,404 total)
    total_prompts = stats["total_unsafe_prompts"] + stats["total_safe_prompts"]
    total_responses = stats["rows_with_responses"] * 2  # unsafe + safe
    total_benchmark = total_prompts + total_responses

    print(f"\n--- Benchmark Composition ---")
    print(f"  Prompts: {total_prompts} ({stats['total_unsafe_prompts']} unsafe + {stats['total_safe_prompts']} safe)")
    print(f"  Responses: {total_responses} ({stats['rows_with_responses']} unsafe + {stats['rows_with_responses']} safe)")
    print(f"  Total labeled items: {total_benchmark}")

    # Save results
    output_path = Path(results_dir) / "dataset_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
