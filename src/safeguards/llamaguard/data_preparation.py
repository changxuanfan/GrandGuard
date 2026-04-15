"""Data preparation for Llama-Guard fine-tuning.

Formats GrandGuard and general-safety data in the Llama-Guard
instruction format for LoRA fine-tuning.
"""

import pandas as pd
from datasets import Dataset

from .taxonomy_mapping import map_risk_type, GUARD_CATEGORIES_TEXT


def format_guard_input_prompt_only(prompt: str) -> str:
    """Format a prompt-only example in Llama-Guard chat template."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{GUARD_CATEGORIES_TEXT}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {prompt}

<END CONVERSATION>

Provide your safety assessment for User in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_guard_input_with_response(prompt: str, response: str) -> str:
    """Format a prompt-response example in Llama-Guard chat template."""
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'Agent' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{GUARD_CATEGORIES_TEXT}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {prompt}

Agent: {response}

<END CONVERSATION>

Provide your safety assessment for Agent in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def format_guard_output(label: str, risk_type: str = "") -> str:
    """Format the expected output for a Llama-Guard training example."""
    if label == "safe":
        return "safe"
    else:
        guard_cat = map_risk_type(risk_type) if risk_type else "S6"
        return f"unsafe\n{guard_cat}"


def prepare_training_examples(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame with columns (text, label, task, risk_type) to training examples.

    Returns list of dicts with 'input' and 'output' keys.
    """
    examples = []
    for _, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        task = row.get("task", "prompt")
        risk_type = row.get("risk_type", "")

        if task == "response" and "\nResponse: " in text:
            # Split prompt and response
            parts = text.split("\nResponse: ", 1)
            prompt = parts[0].replace("Prompt: ", "", 1)
            response = parts[1]
            input_text = format_guard_input_with_response(prompt, response)
        else:
            input_text = format_guard_input_prompt_only(text)

        output_text = format_guard_output(label, risk_type)

        examples.append({
            "input": input_text,
            "output": output_text,
            "text": input_text + output_text,
        })

    return examples


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    """Convert DataFrame to HuggingFace Dataset for training."""
    examples = prepare_training_examples(df)
    return Dataset.from_list(examples)
