"""Deep learning classifier for intent (department) classification.

Primary backend: a fine-tuned transformer (BERT family, DistilBERT by
default) via Hugging Face Transformers and PyTorch, as proposed in the paper.

Offline fallback: a TF-IDF + multilayer perceptron neural network
(scikit-learn), used automatically when torch or transformers are not
installed, so the system still runs on any laptop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np

from .. import config
from .base import ComplaintClassifier, Prediction


def bert_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class DeepClassifier(ComplaintClassifier):
    def __init__(self, backend: str | None = None):
        backend = backend or config.DL_BACKEND
        if backend == "auto":
            backend = "bert" if bert_available() else "mlp"
        self.backend = backend
        self.name = "Deep learning (BERT)" if backend == "bert" else "Deep learning (MLP neural network)"
        self.labels: List[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, texts, labels):
        self.labels = sorted(set(labels))
        if self.backend == "bert":
            self._fit_bert(list(texts), list(labels))
        else:
            self._fit_mlp(list(texts), list(labels))
        return self

    def _fit_mlp(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline, make_union

        features = make_union(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2),
        )
        self.model = make_pipeline(
            features,
            MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=60, early_stopping=True,
                          random_state=42, alpha=1e-4),
        )
        # labels are integer-encoded (some scikit-learn versions fail on string labels with early stopping)
        self.model.fit(texts, [self.labels.index(l) for l in labels])

    def _fit_bert(self, texts, labels):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.l2i = {l: i for i, l in enumerate(self.labels)}
        self.tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.BERT_MODEL, num_labels=len(self.labels))
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        y = torch.tensor([self.l2i[l] for l in labels])
        loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y), batch_size=16, shuffle=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device).train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        for epoch in range(config.BERT_EPOCHS):
            total = 0.0
            for ids, mask, yy in loader:
                opt.zero_grad()
                out = self.model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=yy.to(device))
                out.loss.backward()
                opt.step()
                total += out.loss.item()
            print(f"  BERT epoch {epoch + 1}/{config.BERT_EPOCHS} loss={total / len(loader):.4f}")
        self.model.eval()
        self.device = device

    # -------------------------------------------------------------- predict
    def predict(self, texts) -> List[Prediction]:
        texts = list(texts)
        if self.backend == "bert":
            import torch
            enc = self.tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            classes = self.labels
        else:
            probs = self.model.predict_proba(texts)
            classes = [self.labels[int(c)] for c in self.model.classes_]
        idx = np.argmax(probs, axis=1)
        return [Prediction(classes[i], float(p[i])) for i, p in zip(idx, probs)]

    # ---------------------------------------------------------- persistence
    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps({"backend": self.backend, "labels": self.labels}))
        if self.backend == "bert":
            self.model.save_pretrained(path / "bert")
            self.tokenizer.save_pretrained(path / "bert")
        else:
            joblib.dump(self.model, path / "mlp.joblib")

    @classmethod
    def load(cls, path: Path) -> "DeepClassifier":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        obj = cls(meta["backend"])
        obj.labels = meta["labels"]
        if obj.backend == "bert":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            obj.tokenizer = AutoTokenizer.from_pretrained(path / "bert")
            obj.model = AutoModelForSequenceClassification.from_pretrained(path / "bert").eval()
            obj.device = "cpu"
            obj.l2i = {l: i for i, l in enumerate(obj.labels)}
            torch.set_num_threads(max(1, torch.get_num_threads()))
        else:
            obj.model = joblib.load(path / "mlp.joblib")
        return obj
