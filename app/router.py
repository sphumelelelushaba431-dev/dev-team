"""Department routing, urgency classification and human-in-the-loop escalation."""
from __future__ import annotations

import re

from . import config

URGENCY_RULES = [
    ("Critical", re.compile(r"\b(suicid\w*|kill myself|self[- ]harm|assault\w*|rape\w*|stabbed|robbed|fire|"
                            r"gas leak|electrocut\w*|emergency|unsafe|threat\w*)\b", re.I)),
    ("High", re.compile(r"\b(urgent\w*|asap|immediately|today|tomorrow|deadline|exam (is )?tomorrow|"
                        r"no (water|electricity|power)|locked out|blocked|deregistered|cannot register|"
                        r"can't register|harass\w*|flood\w*|for weeks|still not|can't buy food|cannot buy food|no food|"
                        r"hungry|starving|evicted|nowhere to sleep|wheelchair|disabilit\w*)\b", re.I)),
    ("Medium", re.compile(r"\b(broken|not working|delay\w*|late|wrong|missing|error|problem|issue|"
                          r"failed|unfair|leak\w*)\b", re.I)),
]

SLA_HOURS = {"Critical": 4, "High": 24, "Medium": 72, "Low": 120}


def classify_urgency(text: str) -> str:
    for level, pattern in URGENCY_RULES:
        if pattern.search(text):
            return level
    return "Low"


def decide_status(confidence: float, urgency: str, guard: dict) -> tuple[str, str]:
    """Return (status, reason). Escalate whenever a human should look at it."""
    if guard.get("injection"):
        return "Escalated", "Possible misuse (prompt injection) flagged for review"
    if guard.get("safety") or urgency == "Critical":
        return "Escalated", "Safety-related message: referred to staff immediately"
    if guard.get("formal"):
        return "Escalated", "Formal complaint lodged"
    if confidence < config.CONFIDENCE_THRESHOLD:
        return "Escalated", f"Low classification confidence ({confidence:.2f})"
    return "Open", "Routed automatically"
