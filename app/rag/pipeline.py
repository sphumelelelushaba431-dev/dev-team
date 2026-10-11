"""RAG answer pipeline: rewrite -> retrieve -> generate -> guardrail check."""
from __future__ import annotations

import re
import time

from .. import config
from .components import ConversationMemory, Guardrails, QueryRewriter
from .embedder import load_embedder
from .vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are CampusVoice, the student support assistant of a university. Answer ONLY from the "
    "policy extracts provided. If the extracts do not contain the answer, say you will pass the "
    "query to the relevant office. Be brief (at most 5 sentences), friendly and specific. Mention "
    "contact points, deadlines or turnaround times when the extracts give them. Do not cite the "
    "extracts: no bracketed numbers, file names or section titles in the answer."
)


SYNONYMS = {"cost": {"fee", "fees", "charge"}, "price": {"fee", "fees"}, "pay": {"payment", "fee"},
            "wifi": {"network", "connection"}, "broken": {"repair", "repairs", "maintenance"},
            "money": {"allowance", "allowances", "payment"}, "res": {"residence", "residences"},
            "long": {"within", "days"}, "when": {"within", "days", "date"}}


class RAGPipeline:
    def __init__(self, llm=None, kb_prefix=None):
        kb_prefix = kb_prefix or config.ARTIFACTS_DIR / "kb"
        self.store = VectorStore.load(kb_prefix)
        self.embedder = load_embedder(kb_prefix)
        self.llm = llm
        self.memory = ConversationMemory()
        self.rewriter = QueryRewriter(llm)
        self.guardrails = Guardrails()

    def retrieve(self, query: str, k: int = None, department: str | None = None):
        hits = self.store.search(self.embedder.encode([query])[0], k=(k or config.TOP_K) * 3)
        if department:  # prefer the routed department's documents, keep general policy
            preferred = [h for h in hits if h["department"] in (department, "General")]
            hits = preferred + [h for h in hits if h not in preferred]
        return [h for h in hits if h["score"] >= config.MIN_RETRIEVAL_SCORE][: k or config.TOP_K]

    @staticmethod
    def extractive_answer(query: str, hits, department: str | None = None) -> str:
        """No-LLM fallback: pick the sentences that best overlap with the query."""
        qwords = set(re.findall(r"[a-z]{3,}", query.lower()))
        for word, synonyms in SYNONYMS.items():
            if word in qwords:
                qwords |= synonyms
        scored = []
        for rank, h in enumerate(hits):
            for sent in re.split(r"(?<=[.!?])\s+", h["text"]):
                words = set(re.findall(r"[a-z]{3,}", sent.lower()))
                if len(words) < 6:  # skip headings and fragments
                    continue
                bonus = 1.5 if department and h.get("department") == department else 0.0
                scored.append((len(qwords & words) + 2 * h["score"] - 0.3 * rank + bonus, sent.strip()))
        best = [s for _, s in sorted(scored, key=lambda x: -x[0])[:3]]
        return " ".join(best)

    def answer(self, text: str, session_id: str = "default", department: str | None = None) -> dict:
        t0 = time.perf_counter()
        checks = self.guardrails.check_input(text)
        if checks["injection"]:
            return {"answer": "I can only help with student support questions and complaints.",
                    "sources": [], "query": text, "grounded": False, "guardrails": checks,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "generator": "guardrail"}
        query = self.rewriter.rewrite(checks["text"], self.memory, session_id)
        hits = self.retrieve(query, department=department)
        contexts = [h["text"] for h in hits]
        generator = "extractive"
        if not hits:
            answer = ("I could not find this in the university's policies, so I have passed your "
                      "message to the relevant office. You will receive a reference number.")
        elif self.llm is not None and self.llm.available:
            extracts = "\n\n".join(f"[{i + 1}] ({h['source']} - {h['section']}) {h['text']}" for i, h in enumerate(hits))
            history = self.memory.as_text(session_id)
            prompt = (f"Conversation so far:\n{history or '(none)'}\n\nPolicy extracts:\n{extracts}\n\n"
                      f"Student message: {query}\n\nAnswer:")
            try:
                answer = self.llm.generate(SYSTEM_PROMPT, prompt, max_tokens=300)
                generator = self.llm.label
            except Exception as exc:  # network or quota problem: degrade gracefully
                answer = self.extractive_answer(query, hits, department)
                generator = f"extractive (LLM error: {type(exc).__name__})"
        else:
            answer = self.extractive_answer(query, hits, department)
        grounded = bool(hits) and self.guardrails.grounded(answer, contexts)
        self.memory.add(session_id, "user", checks["text"])
        self.memory.add(session_id, "assistant", answer)
        return {
            "answer": answer, "query": query, "grounded": grounded, "guardrails": checks,
            "sources": [{"source": h["source"], "section": h["section"], "score": round(h["score"], 3)} for h in hits],
            "generator": generator, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
