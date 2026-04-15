#!/usr/bin/env python3
"""Script 09: Fine-tune Llama-Guard-3 with LoRA.

Prepares the mixed training data (62% GrandGuard + 38% general safety)
and fine-tunes Llama-Guard-3-8B with LoRA for elderly-specific safety
classification.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.loader import load_dataset
from src.data.splits import create_guard_splits, mix_with_general_data
from src.data.external import prepare_general_safety_mix
from src.safeguards.llamaguard.data_preparation import prepare_dataset
from src.safeguards.llamaguard.train import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-Guard-3")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-general", action="store_true",
                       help="Skip loading external general-safety data")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    training_config = config.training

    print("=" * 60)
    print("Llama-Guard-3 Fine-Tuning Pipeline")
    print("=" * 60)

    # Step 1: Load GrandGuard dataset
    print("\n[1/5] Loading GrandGuard dataset...")
    df = load_dataset(config.dataset_path)
    print(f"  Loaded {len(df)} rows")

    # Step 2: Create train/eval splits
    print("\n[2/5] Creating train/eval splits...")
    train_gg, eval_gg = create_guard_splits(df, seed=config.seed)
    print(f"  GrandGuard train: {len(train_gg)} examples")
    print(f"  GrandGuard eval: {len(eval_gg)} examples")

    # Step 3: Load general safety data
    if not args.skip_general:
        print("\n[3/5] Loading general safety data (AILuminate, ToxicChat, XSTest)...")
        try:
            general_df = prepare_general_safety_mix(n_samples=1242)
            print(f"  General safety: {len(general_df)} examples")

            # Mix with GrandGuard data
            train_mixed = mix_with_general_data(
                train_gg, general_df,
                grandguard_fraction=training_config.grandguard_fraction,
            )
            print(f"  Mixed train set: {len(train_mixed)} examples "
                  f"({training_config.grandguard_fraction:.0%} GrandGuard)")
        except Exception as e:
            print(f"  Warning: Could not load general data: {e}")
            print("  Using GrandGuard data only")
            train_mixed = train_gg
    else:
        print("\n[3/5] Skipping general safety data")
        train_mixed = train_gg

    # Step 4: Prepare HuggingFace datasets
    print("\n[4/5] Preparing training datasets...")
    train_dataset = prepare_dataset(train_mixed)
    eval_dataset = prepare_dataset(eval_gg)
    print(f"  Train dataset: {len(train_dataset)} examples")
    print(f"  Eval dataset: {len(eval_dataset)} examples")

    # Step 5: Run training
    print("\n[5/5] Starting LoRA fine-tuning...")
    print(f"  Base model: {training_config.base_model}")
    print(f"  LoRA rank: {training_config.lora.rank}")
    print(f"  Max steps: {training_config.schedule.max_steps}")
    print(f"  Learning rate: {training_config.schedule.learning_rate}")

    adapter_path = run_training_pipeline(
        train_dataset, eval_dataset, training_config,
    )

    print(f"\nTraining complete! Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()
