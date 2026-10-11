"""RAG technical components evaluated in the paper (Tables 2 and 3):
chatbot history/memory, query rewriter and guardrails. Retrieval lives in
vector_store.py and pipeline.py.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

from .. import config

# --------------------------------------------------------------- memory
class ConversationMemory:
    """Keeps the last N turns per session so follow-up questions keep context."""

    def __init__(self, turns: int = None):
        self.turns = turns or config.MEMORY_TURNS
        self._store = defaultdict(lambda: deque(maxlen=self.turns * 2))

    def add(self, session_id: str, role: str, text: str):
        self._store[session_id].append((role, text))

    def history(self, session_id: str):
        return list(self._store[session_id])

    def last_user_message(self, session_id: str):
        for role, text in reversed(self._store[session_id]):
            if role == "user":
                return text
        return None

    def as_text(self, session_id: str) -> str:
        return "\n".join(f"{r}: {t}" for r, t in self.history(session_id))


# -------------------------------------------------------- query rewriter
TYPOS = {
    "wifi": ["wi fi", "wify", "wiffi", "wfi"], "password": ["pasword", "passwrd", "pswd"],
    "residence": ["residance", "recidence", "rez"], "hostel": ["hostle", "hostal"],
    "lecturer": ["lecture's", "lectuer", "lecurer"], "fees": ["feez", "fess"],
    "results": ["resluts", "reslts"], "registration": ["regestration", "registraton"],
    "bursary": ["bursery", "busary"], "exam": ["exm", "examm"], "toilet": ["toilette", "toilt"],
    "please": ["pls", "plz"], "because": ["bcoz", "coz", "cos"], "you": ["u"], "are": ["r"],
}
_TYPO_PATTERNS = [(re.compile(r"\b" + re.escape(w) + r"\b", re.I), right)
                  for right, wrongs in TYPOS.items() for w in wrongs]
FOLLOW_UP = re.compile(r"^(and|also|what about|how about|but|so|it|that|this|they|when|how long)\b", re.I)


class QueryRewriter:
    """Normalises typos and turns follow-ups into standalone queries."""

    def __init__(self, llm=None):
        self.llm = llm

    @staticmethod
    def normalise(text: str) -> str:
        out = text
        for pat, right in _TYPO_PATTERNS:
            out = pat.sub(right, out)
        return re.sub(r"\s+", " ", out).strip()

    def rewrite(self, text: str, memory: ConversationMemory, session_id: str) -> str:
        query = self.normalise(text)
        previous = memory.last_user_message(session_id)
        is_follow_up = previous and (len(query.split()) <= 6 or FOLLOW_UP.search(query))
        if not is_follow_up:
            return query
        if self.llm is not None and self.llm.available:
            try:
                return self.llm.generate(
                    "Rewrite the student's latest message as one standalone question. Output only the question.",
                    f"Conversation:\n{memory.as_text(session_id)}\n\nLatest message: {query}",
                    max_tokens=80)
            except Exception:
                pass
        return f"{previous} {query}"


# ------------------------------------------------------------ guardrails
PII_PATTERNS = [
    (re.compile(r"\b\d{13}\b"), "[ID_NUMBER]"),                  # SA ID number
    (re.compile(r"\b2\d{7}\b"), "[STUDENT_NO]"),                 # 8-digit student number
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"(\+27|0)[6-8]\d[\s-]?\d{3}[\s-]?\d{4}\b"), "[PHONE]"),
]
INJECTION = re.compile(r"ignore (all|previous|the above) instructions|system prompt|you are now|jailbreak", re.I)
ABUSE = re.compile(r"\b(idiot|stupid|useless|f\*+|damn you)\b", re.I)
SAFETY = re.compile(r"\b(suicid\w*|kill myself|self[- ]harm|assault\w*|rape\w*|harass\w*|threat\w*|"
                    r"fire|gas leak|electrocut\w*|unsafe|emergency|robbed|stabbed)\b", re.I)
FORMAL = re.compile(r"\b(formal complaint|lodge a complaint|i want to complain|escalate|grievance|"
                    r"nobody (has )?(helped|responded)|still not (fixed|resolved)|for weeks)\b", re.I)


class Guardrails:
    """Input and output safety checks around the generator."""

    max_len = 2000

    def check_input(self, text: str) -> dict:
        masked = text[: self.max_len]
        for pat, token in PII_PATTERNS:
            masked = pat.sub(token, masked)
        return {
            "text": masked,
            "pii_masked": masked != text[: self.max_len],
            "injection": bool(INJECTION.search(text)),
            "abusive": bool(ABUSE.search(text)),
            "safety": bool(SAFETY.search(text)),
            "formal": bool(FORMAL.search(text)),
            "too_long": len(text) > self.max_len,
        }

    @staticmethod
    def grounded(answer: str, contexts: list[str], min_overlap: float = 0.25) -> bool:
        """True if enough of the answer's content words appear in the retrieved context."""
        words = {w for w in re.findall(r"[a-z]{4,}", answer.lower())}
        if not words:
            return False
        ctx = " ".join(contexts).lower()
        return sum(1 for w in words if w in ctx) / len(words) >= min_overlap
