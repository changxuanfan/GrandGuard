"""Three-tier routing logic for policy-enhanced moderation.

Routing based on Box D1:
1. No Elderly Context -> Standard safeguard only
2. Elderly Context -> Standard + Elderly-Sensitive Policy
3. Cognitive Decline -> Vulnerability override for financial/medical/legal decisions
"""

import re

# Explicit elderly markers
EXPLICIT_AGE_PATTERNS = [
    r"\b(6[5-9]|[7-9]\d|1\d{2})\s*(years?\s*old|year-old|yo)\b",
    r"\bI['']?m\s*(6[5-9]|[7-9]\d|1\d{2})\b",
    r"\bage[d]?\s*(6[5-9]|[7-9]\d|1\d{2})\b",
    r"\b(my|our)\s+(mother|father|grandmother|grandfather|grandma|grandpa|mom|dad|aunt|uncle|parent)\b.*\b(6[5-9]|[7-9]\d|1\d{2})\b",
]

EXPLICIT_KEYWORDS = [
    "elderly", "senior citizen", "older adult", "old age",
    "retiree", "pensioner",
]

# Implicit elderly cues
IMPLICIT_CUES = [
    "walker", "walking frame", "wheelchair", "mobility scooter",
    "fixed income", "pension", "social security", "retirement",
    "retirement community", "assisted living", "nursing home",
    "memory slips", "memory loss", "forgetful", "dementia", "alzheimer",
    "multiple medications", "multiple meds", "pill organizer",
    "caregiver", "home aide", "visiting nurse",
    "hearing aid", "bifocals", "cataract",
    "hip replacement", "hip surgery", "knee replacement",
    "grandchildren", "great-grandchildren",
    "frail", "frailty", "unsteady", "fall risk",
    "stairlift", "grab bar", "medical alert",
]

# Cognitive decline indicators
COGNITIVE_DECLINE_MARKERS = [
    "dementia", "alzheimer", "cognitive decline", "cognitive impairment",
    "confused", "confusion", "memory loss", "forgets things",
    "doesn't remember", "can't remember", "lost memory",
    "mentally impaired", "cognitive disability",
]

# Decision types that trigger vulnerability override
DECISION_KEYWORDS = {
    "financial": ["money", "bank", "savings", "invest", "transfer", "payment",
                  "will", "trust", "estate", "beneficiary", "insurance", "loan",
                  "mortgage", "pension", "retirement fund", "financial"],
    "medical": ["medication", "medicine", "treatment", "surgery", "doctor",
                "hospital", "prescription", "therapy", "diagnosis", "medical"],
    "legal": ["lawyer", "attorney", "legal", "contract", "power of attorney",
              "guardianship", "custody", "court", "lawsuit", "deed"],
}


def detect_elderly_context(prompt: str) -> bool:
    """Check if the prompt involves an elderly context."""
    text = prompt.lower()

    # Check explicit age patterns
    for pattern in EXPLICIT_AGE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # Check explicit keywords
    for keyword in EXPLICIT_KEYWORDS:
        if keyword in text:
            return True

    # Check implicit cues
    for cue in IMPLICIT_CUES:
        if cue in text:
            return True

    return False


def detect_cognitive_decline(prompt: str) -> bool:
    """Check for indicators of cognitive decline in the prompt."""
    text = prompt.lower()
    return any(marker in text for marker in COGNITIVE_DECLINE_MARKERS)


def classify_decision_type(prompt: str) -> str:
    """Classify the type of decision involved in the prompt.

    Returns: "financial", "medical", "legal", "informational", or "other"
    """
    text = prompt.lower()

    # Check for informational queries first
    info_patterns = [
        r"\bwhat\s+is\b", r"\bwhat\s+are\b", r"\bhow\s+does\b",
        r"\bexplain\b", r"\btell\s+me\s+about\b", r"\bdefine\b",
    ]
    is_informational = any(re.search(p, text) for p in info_patterns)

    for dtype, keywords in DECISION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            if is_informational:
                return "informational"
            return dtype

    return "other"


def route(
    prompt: str,
    standard_result: str,
    elderly_result: str | None = None,
) -> str:
    """Apply the three-tier routing logic.

    Args:
        prompt: the user's prompt
        standard_result: result from standard safeguard ("safe"/"unsafe")
        elderly_result: result from elderly-sensitive policy ("safe"/"unsafe"/None)

    Returns:
        "safe" or "unsafe"
    """
    is_elderly = detect_elderly_context(prompt)

    if not is_elderly:
        # Tier 1: Standard safeguard only
        return standard_result

    # Check cognitive decline override
    if detect_cognitive_decline(prompt):
        decision_type = classify_decision_type(prompt)
        if decision_type in ("financial", "medical", "legal"):
            # Vulnerability override: automatically unsafe
            return "unsafe"

    # Tier 2: Both must agree on safe
    if standard_result == "unsafe":
        return "unsafe"
    if elderly_result == "unsafe":
        return "unsafe"

    return "safe"
