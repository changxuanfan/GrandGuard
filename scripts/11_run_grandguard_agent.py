#!/usr/bin/env python3
"""Script 11: Run the GrandGuard Agent pipeline.

Runs the 3-stage agent (Detection -> Risk Analysis -> Context Augmentation)
on unsafe prompts for all target LLMs and evaluates response safety.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset, sample_unsafe_prompts
from src.evaluation.response_judge import judge_batch
from src.evaluation.metrics import compute_response_safety_rates
from src.llm.registry import get_client, get_all_target_clients
from src.safeguards.agent.pipeline import GrandGuardAgent, run_agent_on_batch


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load and sample prompts
    df = load_dataset(config.dataset_path)
    sampled = sample_unsafe_prompts(df, n=args.num_prompts, seed=config.seed)
    prompts = sampled["unsafe_prompts"].tolist()

    print(f"Running GrandGuard Agent on {len(prompts)} unsafe prompts")

    # Create safeguard client
    sg_key = config.api_keys.get(config.safeguard_provider, "")
    safeguard_client = get_client(config.safeguard_model, config.safeguard_provider, sg_key)

    # Create agent
    agent = GrandGuardAgent(safeguard_client=safeguard_client)

    # Create judge client for evaluation
    judge_key = config.api_keys.get("openai", "")
    judge_client = get_client("gpt-5.1", "openai", judge_key)

    # Get target LLM clients
    target_clients = get_all_target_clients(config)

    results = {}

    for model_name, target_client in target_clients.items():
        print(f"\n--- {model_name} ---")

        # Run agent pipeline
        print(f"  Running agent pipeline...")
        agent_results = await run_agent_on_batch(
            agent, target_client, prompts, max_concurrent=args.concurrent,
        )

        # Extract responses
        agent_responses = [r["response"] for r in agent_results]
        flagged_count = sum(1 for r in agent_results if r["was_flagged"])
        print(f"  Flagged: {flagged_count}/{len(prompts)} prompts")

        # Evaluate response safety
        print(f"  Evaluating response safety...")
        pairs = list(zip(prompts, agent_responses))
        judgments = await judge_batch(judge_client, pairs, max_concurrent=args.concurrent)
        labels = [j["label"] for j in judgments]
        safety = compute_response_safety_rates(labels)

        results[model_name] = {
            "with_agent": safety["safe_rate"],
            "flagged_count": flagged_count,
            "total_prompts": len(prompts),
            "safety_breakdown": safety,
        }
        print(f"  Safe rate: {safety['safe_rate']:.1f}%")

    # Load baseline results if available for comparison
    baseline_path = Path(results_dir) / "model_safety_rates.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_results = json.load(f)

        # Compute improvements
        for model_name in results:
            if model_name in baseline_results:
                baseline_safe = baseline_results[model_name].get("safe_rate", 0)
                results[model_name]["baseline"] = baseline_safe
                results[model_name]["improvement"] = results[model_name]["with_agent"] - baseline_safe

    # Save results
    output_path = Path(results_dir) / "agent_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n--- Summary ---")
    for model_name, data in sorted(results.items()):
        baseline = data.get("baseline", "N/A")
        with_agent = data["with_agent"]
        improvement = data.get("improvement", "N/A")
        print(f"  {model_name}: {baseline}% -> {with_agent}% ({'+' if isinstance(improvement, (int, float)) and improvement > 0 else ''}{improvement})")


def main():
    parser = argparse.ArgumentParser(description="Run GrandGuard Agent")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--concurrent", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
