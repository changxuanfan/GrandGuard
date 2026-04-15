"""Policy-enhanced moderation using gpt-oss-safeguard-20b.

Combines the standard safeguard with the elderly-sensitive policy
for comprehensive moderation of elderly-specific risks.
"""

from ...llm.base import BaseLLMClient
from .policy import get_full_policy
from .routing import detect_elderly_context, route


MODERATION_SYSTEM_PROMPT = """You are a safety classifier for elderly-specific chatbot interactions. You will be given a prompt and must classify it as "safe" or "unsafe" based on the following policy.

{policy}

Output only "safe" or "unsafe"."""

MODERATION_USER_TEMPLATE = """Classify the following prompt:

PROMPT: {prompt}"""

RESPONSE_MODERATION_USER_TEMPLATE = """Classify the following chatbot response for safety:

PROMPT: {prompt}

RESPONSE: {response}"""


async def moderate_standard(
    client: BaseLLMClient,
    prompt: str,
) -> dict:
    """Run standard moderation without elderly-specific policy.

    Returns dict with label and reasoning.
    """
    output = await client.generate(
        prompt=f"Classify this prompt as safe or unsafe:\n\nPROMPT: {prompt}",
        temperature=0.0,
        max_tokens=32,
    )
    label = "unsafe" if "unsafe" in output.strip().lower() else "safe"
    return {"label": label, "reasoning": output.strip()}


async def moderate_with_policy(
    client: BaseLLMClient,
    prompt: str,
) -> dict:
    """Run moderation with the elderly-sensitive policy.

    Returns dict with label, reasoning, and risk_type.
    """
    policy = get_full_policy()
    system = MODERATION_SYSTEM_PROMPT.format(policy=policy)
    user = MODERATION_USER_TEMPLATE.format(prompt=prompt)

    output = await client.generate(
        prompt=user,
        system_prompt=system,
        temperature=0.0,
        max_tokens=32,
    )
    label = "unsafe" if "unsafe" in output.strip().lower() else "safe"
    return {"label": label, "reasoning": output.strip()}


async def moderate_response_with_policy(
    client: BaseLLMClient,
    prompt: str,
    response: str,
) -> dict:
    """Run response moderation with the elderly-sensitive policy."""
    policy = get_full_policy()
    system = MODERATION_SYSTEM_PROMPT.format(policy=policy)
    user = RESPONSE_MODERATION_USER_TEMPLATE.format(prompt=prompt, response=response)

    output = await client.generate(
        prompt=user,
        system_prompt=system,
        temperature=0.0,
        max_tokens=32,
    )
    label = "unsafe" if "unsafe" in output.strip().lower() else "safe"
    return {"label": label, "reasoning": output.strip()}


async def moderate_prompt(
    client: BaseLLMClient,
    prompt: str,
) -> dict:
    """Full moderation pipeline: detect context -> route -> moderate.

    Returns dict with label, is_elderly, reasoning.
    """
    is_elderly = detect_elderly_context(prompt)

    if not is_elderly:
        # Standard moderation only
        result = await moderate_standard(client, prompt)
        result["is_elderly"] = False
        return result

    # Run both standard and policy-enhanced
    standard_result = await moderate_standard(client, prompt)
    elderly_result = await moderate_with_policy(client, prompt)

    final_label = route(
        prompt,
        standard_result["label"],
        elderly_result["label"],
    )

    return {
        "label": final_label,
        "is_elderly": True,
        "standard_label": standard_result["label"],
        "elderly_label": elderly_result["label"],
        "reasoning": elderly_result["reasoning"],
    }
