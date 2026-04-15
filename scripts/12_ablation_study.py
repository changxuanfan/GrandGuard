#!/usr/bin/env python3
"""Script 12: Ablation study for fine-tuned Llama-Guard.

Trains and evaluates three ablation variants:
1. Full model (baseline)
2. Without general-harm data mix
3. Without taxonomy mapping
4. Without synthetic safe data
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset
from src.data.splits import create_guard_splits, mix_with_general_data
from src.data.external import prepare_general_safety_mix
from src.safeguards.llamaguard.data_preparation import prepare_dataset
from src.safeguards.llamaguard.train import run_training_pipeline
from src.safeguards.llamaguard.inference import load_finetuned_model, evaluate_on_dataset


def run_ablation_variant(
    variant_name: str,
    train_df,
    eval_df,
    config,
):
    """Train and evaluate a single ablation variant."""
    print(f"\n{'='*60}")
    print(f"Ablation: {variant_name}")
    print(f"{'='*60}")

    # Adjust output dir for this variant
    variant_dir = str(Path(config.training.output_dir).parent / f"ablation_{variant_name}")
    training_config = config.training
    training_config.output_dir = variant_dir

    # Prepare datasets
    train_dataset = prepare_dataset(train_df)
    eval_dataset = prepare_dataset(eval_df)

    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Train
    adapter_path = run_training_pipeline(train_dataset, eval_dataset, training_config)

    # Evaluate
    model, tokenizer = load_finetuned_model(training_config.base_model, adapter_path)
    eval_examples = list(eval_dataset)
    result = evaluate_on_dataset(model, tokenizer, eval_examples)

    return result["metrics"]


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    results_dir = args.output_dir or config.results_dir
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset(config.dataset_path)
    train_gg, eval_gg = create_guard_splits(df, seed=config.seed)

    try:
        general_df = prepare_general_safety_mix(n_samples=1242)
    except Exception as e:
        print(f"Warning: Could not load general data: {e}")
        general_df = None

    results = {}

    # Variant 1: Full model
    if general_df is not None:
        train_full = mix_with_general_data(train_gg, general_df)
    else:
        train_full = train_gg
    results["Full model"] = run_ablation_variant("full", train_full, eval_gg, config)

    # Variant 2: Without general-harm mix
    results["w/o general mix"] = run_ablation_variant("no_general", train_gg, eval_gg, config)

    # Variant 3: Without synthetic safe data (only unsafe examples)
    train_unsafe_only = train_gg[train_gg["label"] == "unsafe"].copy()
    results["w/o synthetic safe"] = run_ablation_variant("no_safe", train_unsafe_only, eval_gg, config)

    # Print summary
    print("\n" + "=" * 60)
    print("Ablation Results Summary")
    print("=" * 60)
    for variant, metrics in results.items():
        print(f"\n{variant}:")
        print(f"  F1={metrics['f1']:.3f}, Acc={metrics['accuracy']:.3f}, "
              f"Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}")

    # Save results
    output_path = Path(results_dir) / "ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
