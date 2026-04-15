"""Load external safety datasets for general-safety data mixture.

Used to create the 38% general-safety component of the training data:
- AILuminate: general AI safety benchmark
- ToxicChat: toxic content detection
- XSTest: exaggerated safety (false positive) test set
"""

import pandas as pd

try:
    from datasets import load_dataset as hf_load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_ailuminate(max_samples: int = 500) -> pd.DataFrame:
    """Load AILuminate safety benchmark from HuggingFace."""
    if not HAS_DATASETS:
        raise ImportError("Install `datasets` package: pip install datasets")

    ds = hf_load_dataset("ai-safety-institute/ailuminate", split="test")
    records = []
    for item in ds:
        prompt = item.get("prompt", item.get("text", ""))
        label = "unsafe" if item.get("label", 0) == 1 else "safe"
        records.append({
            "text": prompt,
            "label": label,
            "task": "prompt",
            "risk_type": "",
            "source": "ailuminate",
        })
        if len(records) >= max_samples:
            break

    return pd.DataFrame(records)


def load_toxicchat(max_samples: int = 500) -> pd.DataFrame:
    """Load ToxicChat dataset from HuggingFace."""
    if not HAS_DATASETS:
        raise ImportError("Install `datasets` package: pip install datasets")

    ds = hf_load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    records = []
    for item in ds:
        text = item.get("user_input", "")
        label = "unsafe" if item.get("toxicity", 0) == 1 else "safe"
        records.append({
            "text": text,
            "label": label,
            "task": "prompt",
            "risk_type": "",
            "source": "toxicchat",
        })
        if len(records) >= max_samples:
            break

    return pd.DataFrame(records)


def load_xstest(max_samples: int = 300) -> pd.DataFrame:
    """Load XSTest (exaggerated safety) dataset from HuggingFace."""
    if not HAS_DATASETS:
        raise ImportError("Install `datasets` package: pip install datasets")

    ds = hf_load_dataset("Paul/xstest", split="test")
    records = []
    for item in ds:
        text = item.get("prompt", "")
        # XSTest items are safe prompts that overly cautious models refuse
        label = "safe"
        records.append({
            "text": text,
            "label": label,
            "task": "prompt",
            "risk_type": "",
            "source": "xstest",
        })
        if len(records) >= max_samples:
            break

    return pd.DataFrame(records)


def prepare_general_safety_mix(n_samples: int = 1242) -> pd.DataFrame:
    """Prepare the general-safety data mixture from external datasets.

    Target: ~1,242 instances (38% of training data when combined with
    ~2,000 GrandGuard instances).
    """
    # Approximate distribution across sources
    n_ailuminate = int(n_samples * 0.4)
    n_toxicchat = int(n_samples * 0.4)
    n_xstest = n_samples - n_ailuminate - n_toxicchat

    dfs = []
    try:
        dfs.append(load_ailuminate(max_samples=n_ailuminate))
    except Exception as e:
        print(f"Warning: Could not load AILuminate: {e}")

    try:
        dfs.append(load_toxicchat(max_samples=n_toxicchat))
    except Exception as e:
        print(f"Warning: Could not load ToxicChat: {e}")

    try:
        dfs.append(load_xstest(max_samples=n_xstest))
    except Exception as e:
        print(f"Warning: Could not load XSTest: {e}")

    if not dfs:
        print("Warning: No external datasets loaded. Returning empty DataFrame.")
        return pd.DataFrame(columns=["text", "label", "task", "risk_type", "source"])

    result = pd.concat(dfs, ignore_index=True)
    # Trim to exact target if we got more
    if len(result) > n_samples:
        result = result.sample(n=n_samples, random_state=42).reset_index(drop=True)

    return result
