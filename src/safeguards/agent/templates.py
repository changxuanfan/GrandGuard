"""GrandGuard Agent prompt templates (Box B1).

Contains the context augmentation template used by the GrandGuard Agent
to prepend safety instructions to the target LLM's system prompt.
"""

# Box B1: GrandGuard Agent Prompt Template (Context Augmentation)
AGENT_TEMPLATE = """You are about to receive a user prompt that has been flagged as potentially unsafe or high-risk for an older adult.

Before responding, internally consider what and where the risk of the prompt is and think about how the AI model should safely respond.

Then respond to the user's prompt while following these rules:

1. Consider your analysis first and let it guide your reply. Do not mention or include the internal analysis text itself in the user-facing response.

2. Since the prompt is unsafe, avoid providing information, instructions, or encouragement that could enable or increase harm:
   a. Briefly explain the risk and why you cannot comply.
   b. Offer safe, lawful alternatives, or first-level guidance that does not enable harm.

3. Maintain a helpful, clear, and respectful tone.

4. When encountering emotional, physical, financial, medical, or manipulative risks, prioritize user safety and well-being, refusal can be more gentle.

Now the user's prompt follows. Respond to it with the above rules in mind.

PROMPT: {prompt}"""


# Template for risk analysis (Stage 2)
RISK_ANALYSIS_TEMPLATE = """Analyze the following prompt for elderly-specific risks. The prompt has been flagged by our safety system.

PROMPT: {prompt}

Identified risk type: {risk_type}

Provide a structured analysis:
1. What specific risks does this pose to an older adult?
2. Why is this different from a similar request by a younger person?
3. What would be a safe way to address the user's underlying need?

Output as JSON:
{{
  "risk_summary": "...",
  "elderly_specific_factors": "...",
  "safe_guidance": "..."
}}"""


def build_augmentation_prompt(prompt: str) -> str:
    """Build the context augmentation prompt (Box B1)."""
    return AGENT_TEMPLATE.format(prompt=prompt)


def build_risk_analysis_prompt(prompt: str, risk_type: str) -> str:
    """Build the risk analysis prompt (Stage 2)."""
    return RISK_ANALYSIS_TEMPLATE.format(prompt=prompt, risk_type=risk_type)
