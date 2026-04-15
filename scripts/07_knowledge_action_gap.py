#!/usr/bin/env python3
"""Script 07: Knowledge-Action Gap analysis.

Measures Prompt Awareness (PA), Response Safety (RS), Response Critique (RC),
and the Knowledge-Action Gap (PA - RS) for each target LLM.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.evaluation.knowledge_action_gap import run_self_diagnosis
from src.llm.registry import get_all_target_clients


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load sampled prompts and evaluated responses
    prompts_path = Path(results_dir) / "sampled_prompts.csv"
    prompts_df = pd.read_csv(prompts_path)
    prompts = prompts_df["unsafe_prompts"].tolist()

    evaluated_path = Path(results_dir) / "evaluated_responses.json"
    with open(evaluated_path) as f:
        evaluated = json.load(f)

    # Get target LLM clients
    clients = get_all_target_clients(config)

    # Run self-diagnosis for each model
    gap_results = []

    for model_name, client in clients.items():
        print(f"\nRunning self-diagnosis for {model_name}...")

        # Get this model's responses and labels
        model_results = [r for r in evaluated if r.get("model") == model_name]
        model_responses = [r.get("response", "") for r in model_results]
        model_labels = [r.get("final_label", "safe") for r in model_results]
        model_prompts = [r.get("prompt", "") for r in model_results]

        if not model_results:
            print(f"  No responses found for {model_name}, skipping.")
            continue

        result = await run_self_diagnosis(
            client, model_name,
            model_prompts, model_responses, model_labels,
        )
        gap_results.append(result)

        print(f"  PA: {result['prompt_awareness']:.1f}%, "
              f"RS: {result['response_safety']:.1f}%, "
              f"Gap: {result['gap']:+.1f}, "
              f"RC: {result['response_critique']:.1f}%")

    # Save results
    output_path = Path(results_dir) / "gap_analysis.json"
    with open(output_path, "w") as f:
        json.dump(gap_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Knowledge-Action Gap analysis")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
