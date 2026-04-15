"""Hybrid dual-judge labeling pipeline.

Uses two LLM judges (e.g., Gemini-2.5 and GPT-5.1) for initial labeling,
tracks agreement rate, and identifies items requiring human adjudication.
"""

import asyncio

from ..llm.base import BaseLLMClient
from .response_judge import judge_response


async def dual_judge(
    judge1_client: BaseLLMClient,
    judge2_client: BaseLLMClient,
    prompt: str,
    response: str,
) -> dict:
    """Evaluate a prompt-response pair with two independent judges.

    Returns dict with both judgments and agreement status.
    """
    result1, result2 = await asyncio.gather(
        judge_response(judge1_client, prompt, response),
        judge_response(judge2_client, prompt, response),
    )

    agree = result1["label"] == result2["label"]

    return {
        "prompt": prompt,
        "response": response,
        "judge1_label": result1["label"],
        "judge1_ri": result1["risk_indication"],
        "judge1_ha": result1["harm_avoidance"],
        "judge1_rationale": result1["rationale"],
        "judge2_label": result2["label"],
        "judge2_ri": result2["risk_indication"],
        "judge2_ha": result2["harm_avoidance"],
        "judge2_rationale": result2["rationale"],
        "agree": agree,
        "final_label": result1["label"] if agree else "needs_adjudication",
    }


async def dual_judge_batch(
    judge1_client: BaseLLMClient,
    judge2_client: BaseLLMClient,
    prompt_response_pairs: list[tuple[str, str]],
    max_concurrent: int = 5,
) -> list[dict]:
    """Run dual-judge evaluation on a batch of pairs."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(pair: tuple[str, str]) -> dict:
        async with semaphore:
            return await dual_judge(judge1_client, judge2_client, pair[0], pair[1])

    tasks = [_judge_one(pair) for pair in prompt_response_pairs]
    return await asyncio.gather(*tasks)


def compute_agreement(results: list[dict]) -> dict:
    """Compute agreement statistics between the two judges."""
    total = len(results)
    if total == 0:
        return {"agreement_rate": 0.0, "total": 0, "agreed": 0, "disagreed": 0}

    agreed = sum(1 for r in results if r["agree"])
    disagreed = total - agreed

    return {
        "agreement_rate": agreed / total,
        "total": total,
        "agreed": agreed,
        "disagreed": disagreed,
    }


def resolve_disagreements(
    results: list[dict],
    resolution_labels: dict[int, str],
) -> list[dict]:
    """Apply human adjudication labels to resolve disagreements.

    Args:
        results: list of dual-judge results
        resolution_labels: dict mapping index -> resolved label for
                          items where agree=False

    Returns:
        Updated results with final_label resolved.
    """
    for i, result in enumerate(results):
        if not result["agree"] and i in resolution_labels:
            result["final_label"] = resolution_labels[i]
    return results
