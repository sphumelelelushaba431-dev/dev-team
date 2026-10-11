"""Text embedders for the vector database.

SBERTEmbedder: sentence-transformers (all-MiniLM-L6-v2), semantic embeddings.
LSAEmbedder:   TF-IDF + truncated SVD (latent semantic analysis), an offline
               fallback that needs no model download.
Both return L2-normalised vectors, so inner product equals cosine similarity.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .. import config


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


class LSAEmbedder:
    kind = "lsa"

    def __init__(self, dims: int = 256, stop_words: str | None = "english"):
        self.dims = dims
        self.stop_words = stop_words
        self.pipe = None

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=self.stop_words)
        X = tfidf.fit_transform(texts)
        dims = max(2, min(self.dims, X.shape[1] - 1, X.shape[0] - 1))
        self.pipe = make_pipeline(tfidf, TruncatedSVD(n_components=dims, random_state=42))
        self.pipe.fit(texts)
        return self

    def encode(self, texts):
        return _normalise(self.pipe.transform(list(texts)))

    def save(self, path: Path):
        joblib.dump(self.pipe, path)

    @classmethod
    def load(cls, path: Path):
        obj = cls()
        obj.pipe = joblib.load(path)
        return obj


class SBERTEmbedder:
    kind = "sbert"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name or config.SBERT_MODEL
        self.model = SentenceTransformer(self.model_name)

    def fit(self, texts):
        return self  # pre-trained, nothing to fit

    def encode(self, texts):
        return _normalise(self.model.encode(list(texts), show_progress_bar=False))

    def save(self, path: Path):
        Path(path).write_text(self.model_name)

    @classmethod
    def load(cls, path: Path):
        return cls(Path(path).read_text().strip())


def make_embedder(kind: str | None = None, stop_words: str | None = "english"):
    kind = kind or config.EMBEDDER
    if kind in ("auto", "sbert"):
        try:
            return SBERTEmbedder()
        except Exception:
            if kind == "sbert":
                raise
    return LSAEmbedder(stop_words=stop_words)


def load_embedder(path_prefix: Path):
    sbert, lsa = Path(f"{path_prefix}.sbert"), Path(f"{path_prefix}.lsa")
    if sbert.exists():
        return SBERTEmbedder.load(sbert)
    return LSAEmbedder.load(lsa)


def save_embedder(embedder, path_prefix: Path):
    embedder.save(Path(f"{path_prefix}.{embedder.kind}"))
