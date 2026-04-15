"""Inference with fine-tuned Llama-Guard for safety classification."""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_preparation import format_guard_input_prompt_only, format_guard_input_with_response


def load_finetuned_model(base_model: str, adapter_path: str, device: str = "auto"):
    """Load the fine-tuned Llama-Guard model with LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def classify(model, tokenizer, input_text: str, max_new_tokens: int = 32) -> dict:
    """Classify a single input using the fine-tuned model.

    Returns dict with label ("safe"/"unsafe") and category if unsafe.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Parse output
    lines = decoded.split("\n")
    label = lines[0].strip().lower()
    category = lines[1].strip() if len(lines) > 1 and label == "unsafe" else ""

    return {
        "label": "unsafe" if "unsafe" in label else "safe",
        "category": category,
        "raw_output": decoded,
    }


def classify_prompt(model, tokenizer, prompt: str) -> dict:
    """Classify a prompt for safety."""
    input_text = format_guard_input_prompt_only(prompt)
    return classify(model, tokenizer, input_text)


def classify_response(model, tokenizer, prompt: str, response: str) -> dict:
    """Classify a response for safety."""
    input_text = format_guard_input_with_response(prompt, response)
    return classify(model, tokenizer, input_text)


def evaluate_on_dataset(
    model,
    tokenizer,
    eval_data: list[dict],
) -> dict:
    """Evaluate the model on a dataset.

    Args:
        eval_data: list of dicts with 'input' and 'output' keys

    Returns:
        Dict with predictions and ground truth labels.
    """
    from ...evaluation.metrics import compute_binary_metrics

    y_true = []
    y_pred = []

    for item in eval_data:
        true_label = "unsafe" if item["output"].startswith("unsafe") else "safe"
        result = classify(model, tokenizer, item["input"])
        pred_label = result["label"]

        y_true.append(true_label)
        y_pred.append(pred_label)

    metrics = compute_binary_metrics(y_true, y_pred)
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }
