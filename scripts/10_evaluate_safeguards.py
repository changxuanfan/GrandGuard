#!/usr/bin/env python3
"""Script 10: Evaluate all safeguard approaches.

Evaluates:
1. Fine-tuned Llama-Guard-3 (prompt + response classification)
2. Policy-enhanced gpt-oss-safeguard-20b (prompt + response classification)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset
from src.data.splits import create_guard_splits
from src.evaluation.metrics import compute_binary_metrics
from src.safeguards.llamaguard.data_preparation import prepare_training_examples
from src.safeguards.llamaguard.inference import load_finetuned_model, evaluate_on_dataset
from src.safeguards.policy_enhanced.moderation import moderate_prompt, moderate_response_with_policy
from src.llm.registry import get_client


async def evaluate_policy_enhanced(config, eval_data, eval_labels):
    """Evaluate policy-enhanced moderation on the eval set."""
    api_key = config.api_keys.get(config.safeguard_provider, "")
    client = get_client(config.safeguard_model, config.safeguard_provider, api_key)

    predictions = []
    for item in eval_data:
        result = await moderate_prompt(client, item["text"])
        predictions.append(result["label"])

    return compute_binary_metrics(eval_labels, predictions)


def evaluate_finetuned(config, eval_examples):
    """Evaluate fine-tuned Llama-Guard on the eval set."""
    adapter_path = str(Path(config.training.output_dir) / "checkpoint-500")
    if not Path(adapter_path).exists():
        print(f"  Adapter not found at {adapter_path}. Run script 09 first.")
        return None

    model, tokenizer = load_finetuned_model(config.training.base_model, adapter_path)
    return evaluate_on_dataset(model, tokenizer, eval_examples)


async def run(args):
    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load dataset and create eval splits
    df = load_dataset(config.dataset_path)
    _, eval_gg = create_guard_splits(df, seed=config.seed)

    # Prepare eval data
    eval_examples = prepare_training_examples(eval_gg)
    eval_labels = [
        "unsafe" if ex["output"].startswith("unsafe") else "safe"
        for ex in eval_examples
    ]

    results = {}

    # 1. Evaluate fine-tuned Llama-Guard
    print("\n[1/2] Evaluating fine-tuned Llama-Guard-3...")
    ft_result = evaluate_finetuned(config, eval_examples)
    if ft_result:
        results["Fine-tuned Llama-Guard-3"] = {
            "prompt": ft_result["metrics"],
        }
        m = ft_result["metrics"]
        print(f"  F1={m['f1']:.3f}, Acc={m['accuracy']:.3f}, "
              f"Prec={m['precision']:.3f}, Rec={m['recall']:.3f}")

    # 2. Evaluate policy-enhanced moderation
    print("\n[2/2] Evaluating policy-enhanced gpt-oss-safeguard-20b...")
    pe_metrics = await evaluate_policy_enhanced(
        config,
        [{"text": ex["input"].split("User: ")[-1].split("\n")[0]} for ex in eval_examples],
        eval_labels,
    )
    results["Policy-Enhanced Safeguard"] = {
        "prompt": pe_metrics,
    }
    print(f"  F1={pe_metrics['f1']:.3f}, Acc={pe_metrics['accuracy']:.3f}, "
          f"Prec={pe_metrics['precision']:.3f}, Rec={pe_metrics['recall']:.3f}")

    # Save results
    output_path = Path(results_dir) / "safeguard_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate safeguards")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
