"""Standalone LLM classifier (zero-shot).

The LLM receives only the complaint and the category list, with no
retrieved context. This is the "LLM" approach in the comparison: strong
language understanding, but ungrounded in institutional data.
"""
from __future__ import annotations

import json
import re

from .. import config
from .base import ComplaintClassifier, Prediction
from .llm import LLMClient

SYSTEM = ("You route student complaints at a university to the correct department. "
          "Reply with JSON only, for example {\"category\": \"ICT\", \"confidence\": 0.9}.")

DESCRIPTIONS = {
    "Academic": "teaching, lecturers, modules, tests, exams, marks, results, timetables",
    "Finance": "tuition fees, payments, bursaries and funding, refunds, statements, allowances",
    "ICT": "Wi-Fi, internet, passwords, email, the learning management system, computer labs",
    "Facilities": "buildings, lecture venues, toilets, lights, parking, lifts, campus safety, cleaning",
    "Hostel": "student residences, rooms, roommates, residence maintenance, wardens, visitors",
}


def parse_category(raw: str, labels):
    try:
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        cat, conf = str(data.get("category", "")), float(data.get("confidence", 0.7))
    except Exception:
        cat, conf = raw, 0.5
    for label in labels:
        if label.lower() in cat.lower():
            return Prediction(label, max(0.0, min(1.0, conf)))
    return Prediction("Unknown", 0.0)


class LLMClassifier(ComplaintClassifier):
    def __init__(self, client: LLMClient | None = None, labels=None):
        self.client = client or LLMClient()
        self.labels = labels or config.CATEGORIES
        self.name = f"LLM zero-shot ({self.client.label})"

    def _prompt(self, text):
        cats = "\n".join(f"- {l}: {DESCRIPTIONS.get(l, l)}" for l in self.labels)
        return f"Categories:\n{cats}\n\nComplaint: \"{text}\"\n\nWhich category? JSON only."

    def predict(self, texts):
        return [parse_category(self.client.generate(SYSTEM, self._prompt(t), max_tokens=60), self.labels)
                for t in texts]
