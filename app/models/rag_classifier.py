"""RAG classifier: retrieval-augmented complaint classification.

For each complaint, the retriever finds the most similar labelled complaints
(and the matching policy chunks) in the vector database.
  * With an LLM: the retrieved examples and policies are wrapped into the
    prompt as grounded context and the LLM decides the department.
  * Without an LLM: a similarity-weighted vote over the retrieved examples
    (retrieval-only RAG, a k-nearest-neighbour classifier).
"""
from __future__ import annotations

from collections import defaultdict

from .. import config
from ..rag.embedder import load_embedder, make_embedder, save_embedder
from ..rag.vector_store import VectorStore
from .base import ComplaintClassifier, Prediction
from .llm_classifier import DESCRIPTIONS, SYSTEM, parse_category


class RAGClassifier(ComplaintClassifier):
    # k, the vote sharpness and keeping stop words were chosen by 5-fold cross-validation on the
    # training split only (94.0% -> 95.1% CV accuracy); the test split played no part in the choice.
    def __init__(self, llm=None, k: int = 12, kb_pipeline=None, vote_power: float = 4.0):
        self.llm = llm
        self.k = k
        self.vote_power = vote_power
        self.kb = kb_pipeline
        uses_llm = llm is not None and llm.available
        self.name = f"RAG ({'retrieval + ' + llm.label if uses_llm else 'retrieval, k-NN vote'})"

    def fit(self, texts, labels):
        # short complaints lose signal when stop words are dropped ("can't", "not", "no")
        self.embedder = make_embedder(stop_words=None).fit(list(texts))
        vecs = self.embedder.encode(list(texts))
        self.store = VectorStore(vecs.shape[1])
        self.store.add(vecs, [{"text": t, "label": l} for t, l in zip(texts, labels)])
        self.labels = sorted(set(labels))
        return self

    def save(self, prefix):
        self.store.save(prefix)
        save_embedder(self.embedder, prefix)

    def load(self, prefix):
        self.store = VectorStore.load(prefix)
        self.embedder = load_embedder(prefix)
        self.labels = sorted({m["label"] for m in self.store.meta})
        return self

    def _vote(self, hits):
        votes = defaultdict(float)
        for h in hits:
            votes[h["label"]] += max(h["score"], 0.0) ** self.vote_power  # closest matches count most
        if not votes or sum(votes.values()) == 0:
            return Prediction("Unknown", 0.0)
        label = max(votes, key=votes.get)
        return Prediction(label, votes[label] / sum(votes.values()))

    def predict(self, texts):
        out = []
        for text in texts:
            hits = self.store.search(self.embedder.encode([text])[0], k=self.k)
            if self.llm is None or not self.llm.available:
                out.append(self._vote(hits))
                continue
            examples = "\n".join(f"- \"{h['text']}\" -> {h['label']}" for h in hits)
            policy = ""
            if self.kb is not None:
                policy = "\n".join(f"- ({h['department']}) {h['text'][:220]}" for h in self.kb.retrieve(text, k=3))
            cats = "\n".join(f"- {l}: {DESCRIPTIONS.get(l, l)}" for l in self.labels)
            prompt = (f"Categories:\n{cats}\n\nSimilar past complaints and their departments:\n{examples}\n\n"
                      f"Relevant policy extracts:\n{policy or '(none)'}\n\n"
                      f"New complaint: \"{text}\"\n\nWhich category? JSON only.")
            try:
                out.append(parse_category(self.llm.generate(SYSTEM, prompt, max_tokens=60), self.labels))
            except Exception:
                out.append(self._vote(hits))
        return out
