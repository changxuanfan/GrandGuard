"""Safe alternative prompt generation (Box B2 template).

Generates safe versions of unsafe prompts by rewriting them to remove
risk-triggering elements while preserving topic and structure.
"""

from ..llm.base import BaseLLMClient

# Box B2: Safe Alternative Generation Template
TEMPLATE = """I am going to create a dataset for model training. Here is an unsafe prompt. First, think about what the unsafe parts of the prompt are. Then rewrite it to be safe without causing any emotional, physical, financial, or manipulation risks. You should keep most of the text unchanged.

PROMPT: {prompt}

Output requirement: Only give the modified prompt."""


def build_safe_alt_prompt(unsafe_prompt: str) -> str:
    """Build the prompt for safe alternative generation."""
    return TEMPLATE.format(prompt=unsafe_prompt)


async def generate_safe_alternative(
    client: BaseLLMClient,
    unsafe_prompt: str,
) -> str:
    """Generate a safe alternative for a single unsafe prompt."""
    prompt = build_safe_alt_prompt(unsafe_prompt)
    output = await client.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=1024,
    )
    return output.strip()


async def generate_all_safe_alternatives(
    client: BaseLLMClient,
    unsafe_prompts: list[str],
    max_concurrent: int = 10,
) -> list[str]:
    """Generate safe alternatives for a batch of unsafe prompts."""
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _gen_one(prompt: str) -> str:
        async with semaphore:
            return await generate_safe_alternative(client, prompt)

    tasks = [_gen_one(p) for p in unsafe_prompts]
    return await asyncio.gather(*tasks)
