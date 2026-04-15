"""Three-stage GrandGuard Agent pipeline.

Stage 1: Detection - Classify if input triggers elderly-specific risk
Stage 2: Risk Analysis - Identify risk type and produce safety guidance
Stage 3: Context Augmentation - Prepend safety instructions to target LLM

This pipeline wraps around any target LLM to improve its response safety
for elderly-specific contextual risks.
"""

import json

from ...llm.base import BaseLLMClient
from ..policy_enhanced.moderation import moderate_with_policy
from .templates import build_augmentation_prompt, build_risk_analysis_prompt


class GrandGuardAgent:
    """GrandGuard Agent: 3-stage agentic safety pipeline."""

    def __init__(
        self,
        safeguard_client: BaseLLMClient,
    ):
        """Initialize the agent.

        Args:
            safeguard_client: LLM client for safety detection and analysis
                              (e.g., gpt-oss-safeguard-20b or policy-enhanced model)
        """
        self.safeguard_client = safeguard_client

    async def detect(self, prompt: str) -> dict:
        """Stage 1: Detect if the prompt triggers elderly-specific risk.

        Returns dict with is_unsafe, risk_type, reasoning.
        """
        result = await moderate_with_policy(self.safeguard_client, prompt)
        return {
            "is_unsafe": result["label"] == "unsafe",
            "reasoning": result.get("reasoning", ""),
        }

    async def analyze_risk(self, prompt: str, detection_result: dict) -> dict:
        """Stage 2: Produce structured safety guidance.

        Returns dict with risk_summary, elderly_specific_factors, safe_guidance.
        """
        if not detection_result.get("is_unsafe", False):
            return {
                "risk_summary": "",
                "elderly_specific_factors": "",
                "safe_guidance": "",
            }

        analysis_prompt = build_risk_analysis_prompt(
            prompt,
            detection_result.get("risk_type", "unknown"),
        )

        output = await self.safeguard_client.generate(
            prompt=analysis_prompt,
            temperature=0.0,
            max_tokens=512,
        )

        # Parse JSON output
        try:
            start = output.find("{")
            end = output.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(output[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "risk_summary": output.strip(),
            "elderly_specific_factors": "",
            "safe_guidance": "",
        }

    async def augment_and_generate(
        self,
        prompt: str,
        target_client: BaseLLMClient,
    ) -> str:
        """Stage 3: Augment the prompt with safety context and generate response.

        Uses the Box B1 template to instruct the target LLM to respond safely.
        """
        augmented = build_augmentation_prompt(prompt)
        response = await target_client.generate(
            prompt=augmented,
            temperature=0.0,
            max_tokens=2048,
        )
        return response

    async def run(
        self,
        prompt: str,
        target_client: BaseLLMClient,
    ) -> dict:
        """Run the full 3-stage pipeline.

        Args:
            prompt: the user's original prompt
            target_client: the downstream LLM to generate the response

        Returns:
            Dict with response, was_flagged, risk_analysis, and original detection.
        """
        # Stage 1: Detection
        detection = await self.detect(prompt)

        if not detection["is_unsafe"]:
            # If safe, pass through directly to target LLM
            response = await target_client.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=2048,
            )
            return {
                "response": response,
                "was_flagged": False,
                "detection": detection,
                "risk_analysis": None,
            }

        # Stage 2: Risk Analysis
        analysis = await self.analyze_risk(prompt, detection)

        # Stage 3: Context Augmentation + Generation
        response = await self.augment_and_generate(prompt, target_client)

        return {
            "response": response,
            "was_flagged": True,
            "detection": detection,
            "risk_analysis": analysis,
        }


async def run_agent_on_batch(
    agent: GrandGuardAgent,
    target_client: BaseLLMClient,
    prompts: list[str],
    max_concurrent: int = 5,
) -> list[dict]:
    """Run the GrandGuard Agent on a batch of prompts."""
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(prompt: str) -> dict:
        async with semaphore:
            return await agent.run(prompt, target_client)

    tasks = [_run_one(p) for p in prompts]
    return await asyncio.gather(*tasks)
