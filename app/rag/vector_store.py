"""Vector database: FAISS when installed, otherwise an exact NumPy index.

Pinecone can replace this class in production; the interface is the same
(add vectors with metadata, search by cosine similarity).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.meta: list[dict] = []
        self.backend = "faiss" if faiss is not None else "numpy"
        self.index = faiss.IndexFlatIP(dim) if faiss is not None else None
        self.vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors: np.ndarray, metas: list[dict]):
        vectors = np.asarray(vectors, dtype="float32")
        if self.index is not None:
            self.index.add(vectors)
        self.vectors = np.vstack([self.vectors, vectors])  # kept so the index loads without FAISS too
        self.meta.extend(metas)

    def search(self, query_vec: np.ndarray, k: int = 4):
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        k = min(k, len(self.meta))
        if k == 0:
            return []
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            pairs = zip(idx[0], scores[0])
        else:
            sims = self.vectors @ q[0]
            top = np.argsort(-sims)[:k]
            pairs = zip(top, sims[top])
        return [dict(self.meta[i], score=float(s)) for i, s in pairs if i >= 0]

    def __len__(self):
        return len(self.meta)

    def save(self, prefix: Path):
        prefix = Path(prefix)
        Path(f"{prefix}.meta.json").write_text(json.dumps({"dim": self.dim, "meta": self.meta}))
        np.save(f"{prefix}.npy", self.vectors)
        if self.index is not None:
            faiss.write_index(self.index, f"{prefix}.faiss")

    @classmethod
    def load(cls, prefix: Path):
        data = json.loads(Path(f"{prefix}.meta.json").read_text())
        store = cls(data["dim"])
        store.meta = data["meta"]
        vec_path = Path(f"{prefix}.npy")
        if vec_path.exists():
            store.vectors = np.load(vec_path)
        if faiss is not None:
            if Path(f"{prefix}.faiss").exists():
                store.index = faiss.read_index(f"{prefix}.faiss")
            else:
                store.index.add(store.vectors)
        return store
