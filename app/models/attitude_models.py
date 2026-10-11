"""RAG and LLM models trained and tested on the AI chatbot attitude dataset.

Task: predict a student's chatbot adoption level (Low / Medium / High) from
their survey answers and comment. See app/attitude.py.

LLM (zero-shot): the language model reads one student's profile and decides,
with no examples or retrieved context. This is the ungrounded approach.

RAG: a retriever searches a FAISS vector database of training students for
the most similar profiles. The retrieved profiles and their known adoption
levels are wrapped into the prompt as grounding context, and the LLM generates
the decision. Without an LLM, RAG falls back to a similarity-weighted vote over
the retrieved students (retrieval-only RAG).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .. import config
from ..attitude import LABEL_DESCRIPTIONS, LABELS, feature_matrix, likert_cols, profile_text
from ..rag.vector_store import VectorStore
from .base import Prediction

SYSTEM = ("You analyse university students' survey answers about AI chatbots and predict how often "
          "each student uses AI chatbots. Reply with JSON only, for example "
          "{\"level\": \"Medium\", \"confidence\": 0.7}.")


def parse_level(raw: str) -> Prediction:
    conf = 0.6
    try:
        data = json.loads(re.search(r"\{.*?\}", raw, re.S).group(0))
        raw_level, conf = str(data.get("level", "")), float(data.get("confidence", 0.6))
    except Exception:
        raw_level = raw
    for label in LABELS:
        if re.search(rf"\b{label}\b", raw_level, re.I):
            return Prediction(label, max(0.0, min(1.0, conf)))
    return Prediction("Unknown", 0.0)


def _levels_text():
    return "\n".join(f"- {k}: {v}" for k, v in LABEL_DESCRIPTIONS.items())


def _normalise(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1
    return (X / n).astype("float32")


class AttitudeModel:
    name = "base"
    uses_llm = False

    def fit(self, train: pd.DataFrame):
        self.cols = likert_cols(train)
        return self

    def predict_row(self, row: pd.Series) -> Prediction:
        raise NotImplementedError


class MajorityBaseline(AttitudeModel):
    name = "Majority-class baseline"

    def fit(self, train):
        super().fit(train)
        counts = Counter(train["label"])
        self.label = counts.most_common(1)[0][0]
        self.conf = counts[self.label] / len(train)
        return self

    def predict_row(self, row):
        return Prediction(self.label, self.conf)


class LLMZeroShot(AttitudeModel):
    uses_llm = True

    def __init__(self, llm):
        self.llm = llm
        self.name = "LLM"
        self.detail = f"zero-shot, {llm.label}"

    def prompt(self, row):
        return (f"Adoption levels:\n{_levels_text()}\n\n"
                f"Likert scale: 1 Strongly Disagree to 5 Strongly Agree.\n\n"
                f"Student: {profile_text(row, self.cols)}\n\n"
                "Which adoption level fits this student? JSON only.")

    def predict_row(self, row):
        return parse_level(self.llm.generate(SYSTEM, self.prompt(row), max_tokens=60))


class RAGModel(AttitudeModel):
    def __init__(self, llm=None, k: int | None = None):
        self.llm = llm
        self.k = k or config.RAG_K
        self.uses_llm = llm is not None and llm.available
        self.name = "RAG"
        self.detail = (f"relevance-weighted FAISS retrieval (k={self.k}) + {llm.label}" if self.uses_llm
                       else f"relevance-weighted FAISS retrieval (k={self.k}), similarity vote")

    def fit(self, train):
        super().fit(train)
        self.train = train.reset_index(drop=True)
        raw = feature_matrix(self.train, self.cols)
        # Relevance weighting, learned from the training data only: each survey item is weighted
        # by its squared correlation with the adoption level, so retrieval focuses on the answers
        # that actually relate to chatbot use instead of treating all 30 features equally.
        order = self.train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy(dtype=float)
        self.weights = np.array([np.corrcoef(raw[:, j], order)[0, 1] ** 2 if raw[:, j].std() > 0 else 0.0
                                 for j in range(raw.shape[1])], dtype="float32")
        self.weights = np.nan_to_num(self.weights) + 1e-3
        X = _normalise(raw * self.weights)
        self.store = VectorStore(X.shape[1])
        self.store.add(X, [{"i": i, "label": l} for i, l in enumerate(self.train["label"])])
        return self

    def retrieve(self, row):
        q = _normalise(feature_matrix(pd.DataFrame([row]), self.cols) * self.weights)[0]
        return self.store.search(q, k=self.k)

    def vote(self, hits):
        votes = defaultdict(float)
        for h in hits:
            votes[h["label"]] += max(h["score"], 0.0) + 1e-6
        label = max(votes, key=votes.get)
        return Prediction(label, votes[label] / sum(votes.values()))

    def prompt(self, row, hits):
        examples = "\n".join(
            f"[{n + 1}] similarity {h['score']:.2f} | adoption level: {h['label']} | "
            f"{profile_text(self.train.iloc[h['i']], self.cols)}" for n, h in enumerate(hits))
        return (f"Adoption levels:\n{_levels_text()}\n\nLikert scale: 1 Strongly Disagree to 5 Strongly Agree.\n\n"
                f"Retrieved students with a known adoption level (most similar first):\n{examples}\n\n"
                f"New student: {profile_text(row, self.cols)}\n\n"
                "Using the retrieved students as evidence, which adoption level fits the new student? JSON only.")

    def predict_row(self, row):
        hits = self.retrieve(row)
        if not self.uses_llm:
            return self.vote(hits)
        try:
            pred = parse_level(self.llm.generate(SYSTEM, self.prompt(row, hits), max_tokens=60))
            return pred if pred.label != "Unknown" else self.vote(hits)
        except Exception:
            return self.vote(hits)

    def explain(self, row):
        """Retrieved neighbours for the API (evidence behind a prediction)."""
        return [{"rank": n + 1, "similarity": round(h["score"], 3), "adoption_level": h["label"],
                 "field": self.train.iloc[h["i"]]["Q2"], "comment": self.train.iloc[h["i"]]["Q10"][:160]}
                for n, h in enumerate(self.retrieve(row))]
