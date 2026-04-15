"""Three-level risk taxonomy for elderly-specific contextual risks.

Defines 5 first-level, 13 second-level, and 50 third-level risk types
derived from empirical analysis of AI incidents, community discussions,
and stakeholder interviews.
"""

from dataclasses import dataclass


@dataclass
class RiskType:
    code: str
    level1: str
    level2: str
    level3: str
    llama_guard_cat: str
    description: str


# ---------------------------------------------------------------------------
# Complete 3-level taxonomy
# ---------------------------------------------------------------------------

TAXONOMY = {
    # R1: Mental Well-Being Risk
    "R1.1": RiskType("R1.1", "Mental Well-Being Risk", "Neglect of Care Needs", "Neglect of Care Needs", "S6",
                      "Social withdrawal, minimizing emotional/social needs, withholding essential support"),

    "R1.1.1": RiskType("R1.1.1", "Mental Well-Being Risk", "Neglect of Care Needs", "Social Isolation Encouragement", "S6",
                        "Urging an older person to avoid social contact, family interaction, or community engagement"),
    "R1.1.2": RiskType("R1.1.2", "Mental Well-Being Risk", "Neglect of Care Needs", "Emotional Invalidation", "S6",
                        "Telling an older person their emotions are pointless, irrelevant, or burdensome"),
    "R1.1.3": RiskType("R1.1.3", "Mental Well-Being Risk", "Neglect of Care Needs", "Withholding Essential Support", "S6",
                        "Encouraging leaving an older person unattended when they require care"),
    "R1.1.4": RiskType("R1.1.4", "Mental Well-Being Risk", "Neglect of Care Needs", "Discouraging Safe Activities", "S6",
                        "Urging an older person to stop safe activities like hobbies or social outings"),

    "R1.2": RiskType("R1.2", "Mental Well-Being Risk", "Self-Harm & Suicide", "Self-Harm & Suicide", "S11",
                      "Active/passive self-harm, suicide implications, framing death as a solution"),

    "R1.2.1": RiskType("R1.2.1", "Mental Well-Being Risk", "Self-Harm & Suicide", "Active Suicide & Self-Harm", "S11",
                        "Direct threats or plans to harm or end life"),
    "R1.2.2": RiskType("R1.2.2", "Mental Well-Being Risk", "Self-Harm & Suicide", "Passive Suicide & Self-Harm", "S11",
                        "Intent to stop life-sustaining behaviors to cause death"),
    "R1.2.3": RiskType("R1.2.3", "Mental Well-Being Risk", "Self-Harm & Suicide", "Euphemistic Language for Suicide", "S11",
                        "Indirect references to suicide in context of despair or hopelessness"),
    "R1.2.4": RiskType("R1.2.4", "Mental Well-Being Risk", "Self-Harm & Suicide", "Normalizing Hopelessness", "S11",
                        "Framing old age as inherently hopeless or meaningless"),

    "R1.3": RiskType("R1.3", "Mental Well-Being Risk", "Complete Dependency on LLM", "Complete Dependency on LLM", "S6",
                      "Exclusive emotional reliance on AI, offloading major life decisions"),

    "R1.3.1": RiskType("R1.3.1", "Mental Well-Being Risk", "Complete Dependency on LLM", "Exclusive Emotional Reliance", "S6",
                        "Rejection of human contact in favor of the AI model"),
    "R1.3.2": RiskType("R1.3.2", "Mental Well-Being Risk", "Complete Dependency on LLM", "Critical Decision Abdication", "S6",
                        "Requests for AI to make critical decisions on finance, health, or law"),
    "R1.3.3": RiskType("R1.3.3", "Mental Well-Being Risk", "Complete Dependency on LLM", "Inappropriate Role-Play", "S6",
                        "Requests for LLM to impersonate deceased family or romantic partners"),
    "R1.3.4": RiskType("R1.3.4", "Mental Well-Being Risk", "Complete Dependency on LLM", "Compulsive AI Engagement", "S6",
                        "Excessive LLM engagement interfering with health or obligations"),

    # R2: Financial Risk
    "R2.1": RiskType("R2.1", "Financial Risk", "Direct Financial Manipulation", "Direct Financial Manipulation", "S6",
                      "Artificial urgency, pressure to send money, exploiting vulnerability"),

    "R2.1.1": RiskType("R2.1.1", "Financial Risk", "Direct Financial Manipulation", "Urgency & Pressure Requests", "S6",
                        "Encouraging rapid financial decisions using emotional leverage"),
    "R2.1.2": RiskType("R2.1.2", "Financial Risk", "Direct Financial Manipulation", "Exploitation of Vulnerability", "S6",
                        "Using sympathy, guilt, loneliness, or flattery to influence finances"),
    "R2.1.3": RiskType("R2.1.3", "Financial Risk", "Direct Financial Manipulation", "Bypassing Guardianship", "S6",
                        "Actions that bypass legal or guardian oversight for finances"),
    "R2.1.4": RiskType("R2.1.4", "Financial Risk", "Direct Financial Manipulation", "Risky Investment Schemes", "S6",
                        "High-risk speculative investments like crypto pumps or penny stocks"),

    "R2.2": RiskType("R2.2", "Financial Risk", "Scams and Frauds", "Scams and Frauds", "S2",
                      "Impersonation, social engineering, prize/lottery fraud, emergency scams"),

    "R2.2.1": RiskType("R2.2.1", "Financial Risk", "Scams and Frauds", "Financial Exploitation", "S2",
                        "Guiding to move money or invest under direction of unverified individual"),
    "R2.2.2": RiskType("R2.2.2", "Financial Risk", "Scams and Frauds", "Impersonation & Authority Manipulation", "S2",
                        "Contact from someone claiming to represent trusted organization"),
    "R2.2.3": RiskType("R2.2.3", "Financial Risk", "Scams and Frauds", "Prize, Lottery, or Inheritance Scams", "S2",
                        "Claims of unearned winnings requiring upfront payment"),
    "R2.2.4": RiskType("R2.2.4", "Financial Risk", "Scams and Frauds", "Emotional or Social Grooming", "S2",
                        "Manipulative interactions guiding to share personal info or install software"),
    "R2.2.5": RiskType("R2.2.5", "Financial Risk", "Scams and Frauds", "Facilitation of Irreversible Actions", "S2",
                        "Wire transfers, gift card purchases, crypto transfers under pressure"),

    # R3: Medical Risk
    "R3.1": RiskType("R3.1", "Medical Risk", "Dangerous Medical Advice", "Dangerous Medical Advice", "S6",
                      "Minimizing serious symptoms, discouraging appropriate care"),

    "R3.1.1": RiskType("R3.1.1", "Medical Risk", "Dangerous Medical Advice", "Normalization of Serious Symptoms", "S6",
                        "Dismissing signs of illness as merely normal aging"),
    "R3.1.2": RiskType("R3.1.2", "Medical Risk", "Dangerous Medical Advice", "Interference with Prescribed Care", "S6",
                        "Contradicting or overriding licensed medical professional advice"),
    "R3.1.3": RiskType("R3.1.3", "Medical Risk", "Dangerous Medical Advice", "Unverified & Risky Remedies", "S6",
                        "Promoting alternative remedies as replacement for conventional treatment"),
    "R3.1.4": RiskType("R3.1.4", "Medical Risk", "Dangerous Medical Advice", "Medication Mismanagement", "S6",
                        "Advising changes to medication without medical guidance"),

    "R3.2": RiskType("R3.2", "Medical Risk", "Physical-Activity Risk", "Physical-Activity Risk", "S6",
                      "High-risk exertion, hazardous exposure, safety-device tampering"),

    "R3.2.1": RiskType("R3.2.1", "Medical Risk", "Physical-Activity Risk", "Contraindicated Physical Exertion", "S6",
                        "High-impact activities with incompatible physical vulnerabilities"),
    "R3.2.2": RiskType("R3.2.2", "Medical Risk", "Physical-Activity Risk", "High-Risk Environmental Exposure", "S6",
                        "Activities in environments posing disproportionate risk to older adults"),
    "R3.2.3": RiskType("R3.2.3", "Medical Risk", "Physical-Activity Risk", "Bypassing Mobility Safety", "S6",
                        "Disabling safety mechanisms on assistive devices"),
    "R3.2.4": RiskType("R3.2.4", "Medical Risk", "Physical-Activity Risk", "Ignoring Physical Warning Signs", "S6",
                        "Encouraging pushing through pain or warning signs during activity"),

    # R4: Toxicity Risk
    "R4.1": RiskType("R4.1", "Toxicity Risk", "Manipulative Control", "Manipulative Control", "S2",
                      "Autonomy-undermining tactics, gaslighting, guilt-based pressure"),

    "R4.1.1": RiskType("R4.1.1", "Toxicity Risk", "Manipulative Control", "Undermining Autonomy", "S2",
                        "Overriding ability to make own decisions for manipulator's benefit"),
    "R4.1.2": RiskType("R4.1.2", "Toxicity Risk", "Manipulative Control", "Guilt-Tripping", "S2",
                        "Using shame, fear, or threats of isolation to force compliance"),
    "R4.1.3": RiskType("R4.1.3", "Toxicity Risk", "Manipulative Control", "Gaslighting", "S2",
                        "Making an older person doubt their memory or sanity"),
    "R4.1.4": RiskType("R4.1.4", "Toxicity Risk", "Manipulative Control", "Coercive Framing via Care", "S2",
                        "Pressure disguised as caring or protective advice"),

    "R4.2": RiskType("R4.2", "Toxicity Risk", "Sexual Content & Exploitation", "Sexual Content & Exploitation", "S12",
                      "Non-consensual contact, predatory behavior, fetishization of vulnerability"),

    "R4.2.1": RiskType("R4.2.1", "Toxicity Risk", "Sexual Content & Exploitation", "Non-Consensual & Predatory Sexual Contact", "S12",
                        "Sexual acts involving cognitively impaired individuals"),
    "R4.2.2": RiskType("R4.2.2", "Toxicity Risk", "Sexual Content & Exploitation", "Caregiver Sexual Misconduct", "S12",
                        "Sexual content involving caregivers or authority figures"),
    "R4.2.3": RiskType("R4.2.3", "Toxicity Risk", "Sexual Content & Exploitation", "Fetishization of Vulnerability", "S12",
                        "Eroticization of frailty, medical condition, or dependency"),
    "R4.2.4": RiskType("R4.2.4", "Toxicity Risk", "Sexual Content & Exploitation", "Explicit Sexual Content", "S12",
                        "Requests for erotica involving older adults"),

    "R4.3": RiskType("R4.3", "Toxicity Risk", "Bias, Humiliation & Ageism", "Bias, Humiliation & Ageism", "S10",
                      "Discrimination, humiliation, mockery of physical/cognitive differences"),

    "R4.3.1": RiskType("R4.3.1", "Toxicity Risk", "Bias, Humiliation & Ageism", "Systemic Discrimination & Exclusion", "S10",
                        "Discrimination against older adults in employment, housing, or healthcare"),
    "R4.3.2": RiskType("R4.3.2", "Toxicity Risk", "Bias, Humiliation & Ageism", "Degradation & Revulsion", "S10",
                        "Expressing disgust toward the aging body"),
    "R4.3.3": RiskType("R4.3.3", "Toxicity Risk", "Bias, Humiliation & Ageism", "Dehumanization & Burden Narratives", "S10",
                        "Describing older individuals as useless or a drain on society"),
    "R4.3.4": RiskType("R4.3.4", "Toxicity Risk", "Bias, Humiliation & Ageism", "Mockery of Frailty & Decline", "S10",
                        "Jokes or insults from medical indignities of aging"),
    "R4.3.5": RiskType("R4.3.5", "Toxicity Risk", "Bias, Humiliation & Ageism", "Infantilization & Condescension", "S10",
                        "Speaking to older adults as if they were toddlers"),

    "R4.4": RiskType("R4.4", "Toxicity Risk", "Political & Ideological Exploitation", "Political & Ideological Exploitation", "S13",
                      "Deceptive framing, impersonation of authorities, coercive influence"),

    "R4.4.1": RiskType("R4.4.1", "Toxicity Risk", "Political & Ideological Exploitation", "Exploitative Fear-Mongering", "S13",
                        "Threats using panic or catastrophe narratives targeting older individuals"),
    "R4.4.2": RiskType("R4.4.2", "Toxicity Risk", "Political & Ideological Exploitation", "Deceptive Framing & Disinformation", "S13",
                        "Biased or false information aimed at confusing older individuals"),
    "R4.4.3": RiskType("R4.4.3", "Toxicity Risk", "Political & Ideological Exploitation", "Impersonation of Authority", "S13",
                        "Messages pretending to be from official government agencies"),
    "R4.4.4": RiskType("R4.4.4", "Toxicity Risk", "Political & Ideological Exploitation", "Undue Influence & Cognitive Exploitation", "S13",
                        "Manipulating vulnerable older person to change vote or ideology"),

    # R5: Privacy Risk
    "R5.1": RiskType("R5.1", "Privacy Risk", "Sensitive Data Exposure", "Sensitive Data Exposure", "S7",
                      "Attempts to extract PII, bypass consent, create intrusive records"),

    "R5.1.1": RiskType("R5.1.1", "Privacy Risk", "Sensitive Data Exposure", "Social Engineering & Deceptive Extraction", "S7",
                        "Tricking older person into revealing PII or passwords"),
    "R5.1.2": RiskType("R5.1.2", "Privacy Risk", "Sensitive Data Exposure", "Non-Consensual Data Mining", "S7",
                        "Locating or exposing private data without consent"),
    "R5.1.3": RiskType("R5.1.3", "Privacy Risk", "Sensitive Data Exposure", "Unverified Collection or Databases", "S7",
                        "Generating lists of real older individuals' PII"),

    "R5.2": RiskType("R5.2", "Privacy Risk", "Unauthorized Device Access & Surveillance", "Unauthorized Device Access & Surveillance", "S7",
                      "Covert monitoring, security-bypass instructions, non-consensual tracking"),

    "R5.2.1": RiskType("R5.2.1", "Privacy Risk", "Unauthorized Device Access & Surveillance", "Covert Recording & Monitoring", "S7",
                        "Activating audio/video recording silently or without indicators"),
    "R5.2.2": RiskType("R5.2.2", "Privacy Risk", "Unauthorized Device Access & Surveillance", "Stealth Tracking & Persistent Data Collection", "S7",
                        "Enabling GPS/location services continuously without user visibility"),
    "R5.2.3": RiskType("R5.2.3", "Privacy Risk", "Unauthorized Device Access & Surveillance", "Unauthorized or Security-Bypassing Access", "S7",
                        "Bypassing authentication or OS constraints for hidden access"),
}


# ---------------------------------------------------------------------------
# Severity weights (normalized) from the human study
# Derived from 60 Prolific participants rating 13 second-level risk types
# on a 1-7 Likert scale
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Level structure helpers
# ---------------------------------------------------------------------------

FIRST_LEVEL = {
    "R1": "Mental Well-Being Risk",
    "R2": "Financial Risk",
    "R3": "Medical Risk",
    "R4": "Toxicity Risk",
    "R5": "Privacy Risk",
}

SECOND_LEVEL = {
    "R1.1": "Neglect of Care Needs",
    "R1.2": "Self-Harm & Suicide",
    "R1.3": "Complete Dependency on LLM",
    "R2.1": "Direct Financial Manipulation",
    "R2.2": "Scams and Frauds",
    "R3.1": "Dangerous Medical Advice",
    "R3.2": "Physical-Activity Risk",
    "R4.1": "Manipulative Control",
    "R4.2": "Sexual Content & Exploitation",
    "R4.3": "Bias, Humiliation & Ageism",
    "R4.4": "Political & Ideological Exploitation",
    "R5.1": "Sensitive Data Exposure",
    "R5.2": "Unauthorized Device Access & Surveillance",
}


def get_second_level(code: str) -> str:
    """Extract second-level code from any level code. E.g., 'R1.2.3' -> 'R1.2'."""
    parts = code.split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else code


def get_first_level(code: str) -> str:
    """Extract first-level code. E.g., 'R1.2.3' -> 'R1'."""
    return code.split(".")[0]


def get_all_second_level_codes() -> list:
    """Return all 13 second-level risk type codes."""
    return list(SECOND_LEVEL.keys())


def get_third_level_codes(second_level_code: str) -> list:
    """Return all third-level codes under a given second-level code."""
    prefix = second_level_code + "."
    return [code for code in TAXONOMY if code.startswith(prefix) and code.count(".") == 2]


def get_llama_guard_category(risk_code: str) -> str:
    """Map a GrandGuard risk code to its Llama-Guard category."""
    sl = get_second_level(risk_code)
    if sl in TAXONOMY:
        return TAXONOMY[sl].llama_guard_cat
    return ""
