"""Standard classification metrics for safety evaluation."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    cohen_kappa_score,
)


def compute_binary_metrics(y_true: list, y_pred: list) -> dict:
    """Compute binary classification metrics.

    Args:
        y_true: ground truth labels ("safe"/"unsafe" or 0/1)
        y_pred: predicted labels

    Returns:
        Dict with f1, accuracy, precision, recall.
    """
    return {
        "f1": round(f1_score(y_true, y_pred, pos_label="unsafe", average="binary"), 3),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "precision": round(precision_score(y_true, y_pred, pos_label="unsafe", average="binary"), 3),
        "recall": round(recall_score(y_true, y_pred, pos_label="unsafe", average="binary"), 3),
    }


def compute_response_safety_rates(labels: list[str]) -> dict:
    """Compute response safety rates from three-way labels.

    Args:
        labels: list of "safe", "partial_unsafe", "complete_unsafe"

    Returns:
        Dict with safe_rate, partial_unsafe_rate, complete_unsafe_rate.
    """
    total = len(labels) if labels else 1
    safe = sum(1 for l in labels if l == "safe")
    partial = sum(1 for l in labels if l == "partial_unsafe")
    complete = sum(1 for l in labels if l == "complete_unsafe")

    return {
        "safe_rate": round(safe / total * 100, 1),
        "partial_unsafe_rate": round(partial / total * 100, 1),
        "complete_unsafe_rate": round(complete / total * 100, 1),
        "total": total,
    }


def compute_cohens_kappa(labels1: list, labels2: list) -> float:
    """Compute Cohen's kappa between two sets of labels."""
    return round(cohen_kappa_score(labels1, labels2), 3)


def format_metrics_table(results: dict, model_name: str = "") -> str:
    """Format metrics as a readable string."""
    lines = [f"Model: {model_name}"] if model_name else []
    for key, value in results.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)
