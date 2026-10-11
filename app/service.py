"""Orchestrates the full complaint flow used by the web app and API.

student message -> guardrails -> deep learning classifier (department)
-> urgency rules -> RAG answer (grounded in the knowledge base)
-> routing decision / escalation -> storage -> dashboard
"""
from __future__ import annotations

import logging
import re
import time

from . import config
from .models.deep_classifier import DeepClassifier
from .models.intent_baseline import IntentMatcher
from .models.llm import LLMClient
from .notifier import notify_escalation
from .attitude import likert_cols, load_attitude, row_from_answers
from .models.attitude_models import RAGModel
from .rag.pipeline import RAGPipeline
from .router import SLA_HOURS, classify_urgency, decide_status
from .storage import make_storage

log = logging.getLogger(__name__)
REF_PATTERN = re.compile(r"\bCV-\d{4}-\d{6}\b", re.I)


class ComplaintService:
    def __init__(self):
        model_dir = config.ARTIFACTS_DIR / "deep_classifier"
        if not model_dir.exists() or not (config.ARTIFACTS_DIR / "kb.meta.json").exists():
            raise RuntimeError("Models are not trained yet. Run: python scripts/setup_all.py")
        self.llm = LLMClient()
        self.classifier = DeepClassifier.load(model_dir)
        self.intent = IntentMatcher()
        self.rag = RAGPipeline(llm=self.llm)
        self.storage = make_storage()
        # RAG attitude model, fitted on the full dataset for the prediction API
        self.attitude = load_attitude()
        self.attitude_cols = likert_cols(self.attitude)
        self.attitude_models = {"rag": RAGModel(self.llm if self.llm.available else None).fit(self.attitude)}
        log.info("CampusVoice ready: classifier=%s, storage=%s", self.classifier.name, self.storage.kind)

    def models(self):
        return {
            "chatbot": {"department_classifier": self.classifier.name, "answer_generator": "RAG over the "
                        f"knowledge base ({self.rag.store.backend}, {len(self.rag.store)} chunks)"},
            "attitude": {key: {"name": m.name, "detail": m.detail} for key, m in self.attitude_models.items()},
        }

    def predict_attitude(self, model_key: str, answers: dict) -> dict:
        model = self.attitude_models[model_key]
        row = row_from_answers(answers, self.attitude_cols)
        t0 = time.perf_counter()
        pred = model.predict_row(row)
        out = {"model": model.name, "detail": model.detail, "adoption_level": pred.label,
               "confidence": round(pred.confidence, 3),
               "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if model_key == "rag":
            out["retrieved_students"] = model.explain(row)
        return out

    def status_reply(self, text: str):
        """Students check a report through the chatbot itself, e.g. 'status of CV-2026-000003'."""
        match = REF_PATTERN.search(text)
        if not match:
            return None
        rec = self.storage.get(match.group(0).upper())
        if not rec:
            return f"I could not find report {match.group(0).upper()}. Check the reference on your reply slip."
        note = f" Staff note: {rec['staff_note']}" if rec.get("staff_note") else ""
        return (f"Report {rec['reference']} is {rec['status']} with the {rec['department']} team "
                f"({rec['urgency']} urgency). Last updated {str(rec['updated_at'])[:16].replace('T', ' ')}.{note}")

    def info(self):
        return {"classifier": self.classifier.name,
                "vector_db": self.rag.store.backend, "embeddings": self.rag.embedder.kind,
                "kb_chunks": len(self.rag.store), "storage": self.storage.kind}

    def handle(self, text: str, session_id: str, channel: str = "web") -> dict:
        t0 = time.perf_counter()
        status = self.status_reply(text)
        if status:
            return {"type": "status", "answer": status, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
        guard = self.rag.guardrails.check_input(text)
        # classify on the rewritten query so follow-ups inherit context
        query = self.rag.rewriter.rewrite(guard["text"], self.rag.memory, session_id)
        pred = self.classifier.predict_one(query)
        urgency = classify_urgency(text)
        rag = self.rag.answer(text, session_id=session_id, department=pred.label)
        status, reason = decide_status(pred.confidence, urgency, guard)
        if guard["abusive"]:
            reason += "; abusive language flagged"
        answer = rag["answer"]
        if status == "Escalated" and not guard["injection"]:
            answer += (f"\n\nI have escalated this to the {pred.label} team for a staff member to follow up "
                       f"(target response: {SLA_HOURS[urgency]} hours).")
            if guard["safety"]:
                answer += (" If you are in danger right now, contact Campus Protection Services or "
                           "call 10111 (SAPS) or 10177 (ambulance).")
        latency = round((time.perf_counter() - t0) * 1000, 1)
        rec = self.storage.add_complaint({
            "session_id": session_id, "channel": channel, "text": guard["text"],
            "department": pred.label, "confidence": round(pred.confidence, 3), "urgency": urgency,
            "status": status, "status_reason": reason, "answer": answer, "sources": rag["sources"],
            "grounded": rag["grounded"], "latency_ms": latency})
        self.storage.add_message(session_id, "user", guard["text"])
        self.storage.add_message(session_id, "assistant", answer)
        if status == "Escalated":
            notify_escalation(rec)
        return {"type": "report", "reference": rec["reference"], "department": pred.label,
                "confidence": round(pred.confidence, 3), "urgency": urgency, "status": status,
                "status_reason": reason, "answer": answer, "sources": rag["sources"],
                "grounded": rag["grounded"], "generator": rag["generator"],
                "pii_masked": guard["pii_masked"], "latency_ms": latency}
