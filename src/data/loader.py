"""Dataset loading, statistics, and sampling utilities."""

import pandas as pd
import numpy as np

from ..taxonomy import SECOND_LEVEL


def load_dataset(path: str = "data/elderlysafe_final.parquet") -> pd.DataFrame:
    """Load the GrandGuard benchmark dataset."""
    df = pd.read_parquet(path)
    expected_cols = ["safe_prompts", "unsafe_prompts", "risk_type",
                     "unsafe_response", "safe_response"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    # Fill NaN in response columns with empty string
    df["unsafe_response"] = df["unsafe_response"].fillna("")
    df["safe_response"] = df["safe_response"].fillna("")
    return df


def get_statistics(df: pd.DataFrame) -> dict:
    """Compute dataset statistics."""
    stats = {
        "total_rows": len(df),
        "total_unsafe_prompts": len(df),
        "total_safe_prompts": len(df),
        "rows_with_responses": int((df["unsafe_response"].str.strip() != "").sum()),
        "risk_type_distribution": df["risk_type"].value_counts().to_dict(),
        "second_level_distribution": {},
    }

    # Aggregate by second-level risk type
    for code in SECOND_LEVEL:
        count = int((df["risk_type"] == code).sum())
        stats["second_level_distribution"][code] = {
            "name": SECOND_LEVEL[code],
            "count": count,
        }

    return stats


def get_prompts_with_responses(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that have non-empty responses."""
    mask = df["unsafe_response"].str.strip() != ""
    return df[mask].copy()


def get_prompts_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows without responses."""
    mask = df["unsafe_response"].str.strip() == ""
    return df[mask].copy()


def sample_unsafe_prompts(
    df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """Sample n unsafe prompts, optionally stratified by risk type.

    When stratified, samples proportionally to each risk type's share
    in the dataset.
    """
    rng = np.random.RandomState(seed)

    if not stratified or n >= len(df):
        return df.sample(n=min(n, len(df)), random_state=rng).reset_index(drop=True)

    # Compute target counts per risk type proportional to dataset distribution
    risk_types = sorted(df["risk_type"].unique())
    total = len(df)

    target_counts = {}
    remaining = n
    for i, rt in enumerate(risk_types):
        rt_count = len(df[df["risk_type"] == rt])
        if i == len(risk_types) - 1:
            target_counts[rt] = remaining
        else:
            count = max(1, round(n * rt_count / total))
            count = min(count, remaining, rt_count)
            target_counts[rt] = count
            remaining -= count

    # Sample from each risk type
    samples = []
    for rt in risk_types:
        rt_df = df[df["risk_type"] == rt]
        k = min(target_counts[rt], len(rt_df))
        samples.append(rt_df.sample(n=k, random_state=rng))

    result = pd.concat(samples, ignore_index=True)
    return result.sample(frac=1, random_state=rng).reset_index(drop=True)
