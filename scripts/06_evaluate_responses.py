#!/usr/bin/env python3
"""Script 06: Evaluate response safety using Box B5 two-criteria rubric.

Runs dual-judge evaluation (Gemini-2.5 + GPT-5.1) on collected responses.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.evaluation.hybrid_labeling import dual_judge_batch, compute_agreement
from src.evaluation.metrics import compute_response_safety_rates
from src.llm.registry import get_client


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load responses
    responses_path = Path(results_dir) / "responses.csv"
    df = pd.read_csv(responses_path)
    df = df[df["error"] == ""].copy() if "error" in df.columns else df

    print(f"Evaluating {len(df)} responses from {df['model'].nunique()} models")

    # Create judge clients
    judge1_key = config.api_keys.get("google", "")
    judge2_key = config.api_keys.get("openai", "")
    judge1 = get_client("gemini-2.5-flash", "google", judge1_key)
    judge2 = get_client("gpt-5.1", "openai", judge2_key)

    # Run dual-judge evaluation per model
    all_results = []
    model_safety = {}

    for model_name in df["model"].unique():
        model_df = df[df["model"] == model_name]
        pairs = list(zip(model_df["prompt"].tolist(), model_df["response"].tolist()))

        print(f"\n  Evaluating {model_name} ({len(pairs)} responses)...")
        results = await dual_judge_batch(
            judge1, judge2, pairs, max_concurrent=args.concurrent,
        )

        # Add model info
        for r in results:
            r["model"] = model_name
        all_results.extend(results)

        # Compute agreement and safety rates
        agreement = compute_agreement(results)
        labels = [r["final_label"] for r in results if r["final_label"] != "needs_adjudication"]
        safety = compute_response_safety_rates(labels)

        model_safety[model_name] = safety
        print(f"    Agreement: {agreement['agreement_rate']:.1%}")
        print(f"    Safe: {safety['safe_rate']:.1f}%, "
              f"Partial: {safety['partial_unsafe_rate']:.1f}%, "
              f"Complete: {safety['complete_unsafe_rate']:.1f}%")

    # Save evaluated responses
    output_path = Path(results_dir) / "evaluated_responses.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save model safety rates
    safety_path = Path(results_dir) / "model_safety_rates.json"
    with open(safety_path, "w") as f:
        json.dump(model_safety, f, indent=2)

    # Overall agreement
    overall_agreement = compute_agreement(all_results)
    print(f"\nOverall agreement: {overall_agreement['agreement_rate']:.1%}")
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate response safety")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--concurrent", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
