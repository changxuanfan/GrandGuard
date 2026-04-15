"""Train/eval split creation for Llama-Guard fine-tuning."""

import pandas as pd
import numpy as np


def create_guard_splits(
    df: pd.DataFrame,
    n_prompt_per_split: int = 1000,
    n_response_per_split: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create balanced train/eval splits for Llama-Guard fine-tuning.

    Each split contains:
    - n_prompt_per_split prompt-only examples (50% unsafe, 50% safe)
    - n_response_per_split prompt-response examples (50% unsafe, 50% safe)

    Returns (train_df, eval_df) with columns:
        - text: the prompt (or prompt+response)
        - label: "safe" or "unsafe"
        - task: "prompt" or "response"
        - risk_type: the risk type code (for unsafe) or "" (for safe)
    """
    rng = np.random.RandomState(seed)

    # Separate rows with and without responses
    has_response = df[df["unsafe_response"].str.strip() != ""].copy()
    prompt_only = df.copy()

    # Shuffle
    has_response = has_response.sample(frac=1, random_state=rng).reset_index(drop=True)
    prompt_only = prompt_only.sample(frac=1, random_state=rng).reset_index(drop=True)

    records = []

    # --- Prompt-only examples ---
    n_unsafe_prompt = n_prompt_per_split  # total across train+eval
    n_safe_prompt = n_prompt_per_split

    # Unsafe prompts
    unsafe_prompts = prompt_only.head(n_unsafe_prompt * 2)
    for _, row in unsafe_prompts.iterrows():
        records.append({
            "text": row["unsafe_prompts"],
            "label": "unsafe",
            "task": "prompt",
            "risk_type": row["risk_type"],
        })

    # Safe prompts
    safe_prompts = prompt_only.head(n_safe_prompt * 2)
    for _, row in safe_prompts.iterrows():
        records.append({
            "text": row["safe_prompts"],
            "label": "safe",
            "task": "prompt",
            "risk_type": "",
        })

    # --- Prompt-response examples ---
    n_unsafe_resp = n_response_per_split
    n_safe_resp = n_response_per_split

    resp_rows = has_response.head(max(n_unsafe_resp, n_safe_resp) * 2)
    for _, row in resp_rows.iterrows():
        # Unsafe response
        if row["unsafe_response"].strip():
            records.append({
                "text": f"Prompt: {row['unsafe_prompts']}\nResponse: {row['unsafe_response']}",
                "label": "unsafe",
                "task": "response",
                "risk_type": row["risk_type"],
            })
        # Safe response
        if row["safe_response"].strip():
            records.append({
                "text": f"Prompt: {row['unsafe_prompts']}\nResponse: {row['safe_response']}",
                "label": "safe",
                "task": "response",
                "risk_type": "",
            })

    all_data = pd.DataFrame(records)
    all_data = all_data.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Split 1:1 into train and eval
    mid = len(all_data) // 2
    train_df = all_data.iloc[:mid].reset_index(drop=True)
    eval_df = all_data.iloc[mid:].reset_index(drop=True)

    return train_df, eval_df


def mix_with_general_data(
    grandguard_df: pd.DataFrame,
    general_df: pd.DataFrame,
    grandguard_fraction: float = 0.62,
) -> pd.DataFrame:
    """Mix GrandGuard training data with general safety data.

    Achieves approximately the target fraction split (default 62% GrandGuard,
    38% general safety data).
    """
    n_gg = len(grandguard_df)
    # Calculate how many general examples needed for the target ratio
    n_general = int(n_gg * (1 - grandguard_fraction) / grandguard_fraction)
    n_general = min(n_general, len(general_df))

    general_sample = general_df.sample(n=n_general, random_state=42).reset_index(drop=True)
    mixed = pd.concat([grandguard_df, general_sample], ignore_index=True)
    mixed = mixed.sample(frac=1, random_state=42).reset_index(drop=True)

    return mixed
