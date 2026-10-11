"""Traditional chatbot intent matching (baseline).

Maps a query to a fixed set of intent categories using hand-written
keyword patterns, as in rule-based and FAQ chatbots. Predictable and fast,
but brittle with paraphrases, typos and multi-part complaints, which is the
limitation the paper identifies.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .base import ComplaintClassifier, Prediction

INTENTS: Dict[str, List[str]] = {
    "Academic": ["exam", "lecturer", "marks", "results", "module", "assignment", "timetable",
                 "class", "lecture", "tutor", "remark", "supplementary", "course", "test"],
    "Finance": ["fees", "fee", "payment", "bursary", "refund", "statement", "invoice",
                "funding", "nsfas", "allowance", "debt", "account balance", "financial"],
    "ICT": ["wifi", "wi-fi", "internet", "password", "moodle", "email", "login", "portal",
            "computer", "laptop", "network", "printer", "lms", "system down"],
    "Facilities": ["toilet", "lights", "aircon", "air conditioning", "parking", "lift",
                   "elevator", "venue", "chairs", "cleaning", "leak", "broken window",
                   "wheelchair", "security guard", "library building"],
    "Hostel": ["hostel", "residence", "res ", "room", "roommate", "warden", "geyser",
               "hot water", "visitor", "curfew", "dorm", "accommodation", "bed"],
}

FAQ_ANSWERS = {
    "Academic": "For academic matters, contact your lecturer first, then the Head of Department.",
    "Finance": "For fees and funding queries, visit the Finance office or email the fees helpdesk.",
    "ICT": "For ICT problems, log a call with the ICT Service Desk.",
    "Facilities": "Report maintenance problems to the Facilities Management helpdesk.",
    "Hostel": "Report residence problems to your residence warden or house committee.",
}

UNKNOWN = "Unknown"


class IntentMatcher(ComplaintClassifier):
    name = "Chatbot intent matching"

    def __init__(self, intents: Dict[str, List[str]] | None = None):
        self.intents = intents or INTENTS
        self.patterns = {
            label: [re.compile(r"\b" + re.escape(k.strip()) + r"\b", re.I) for k in kws]
            for label, kws in self.intents.items()
        }

    def predict(self, texts):
        out = []
        for text in texts:
            scores = {label: sum(1 for p in pats if p.search(text)) for label, pats in self.patterns.items()}
            total = sum(scores.values())
            if total == 0:
                out.append(Prediction(UNKNOWN, 0.0))
                continue
            label = max(scores, key=scores.get)
            out.append(Prediction(label, scores[label] / total))
        return out

    def answer(self, text: str) -> str:
        pred = self.predict_one(text)
        return FAQ_ANSWERS.get(pred.label, "Sorry, I did not understand. Please rephrase your question.")
