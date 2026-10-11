"""Train the deep learning classifier and build the RAG example index (80% training split)."""
import time

import _path  # noqa: F401
from app import config
from app.data import load_complaints, split
from app.models.deep_classifier import DeepClassifier
from app.models.rag_classifier import RAGClassifier


def main():
    df = load_complaints()
    train, test = split(df)
    print(f"Dataset: {df.attrs['source']} | {len(df)} complaints | {df['label'].nunique()} departments")
    print(f"Train {len(train)} / test {len(test)}")
    t0 = time.time()
    clf = DeepClassifier().fit(train["text"].tolist(), train["label"].tolist())
    clf.save(config.ARTIFACTS_DIR / "deep_classifier")
    print(f"Trained {clf.name} in {time.time() - t0:.1f}s")
    rag = RAGClassifier().fit(train["text"].tolist(), train["label"].tolist())
    rag.save(config.ARTIFACTS_DIR / "rag_examples")
    print(f"Built RAG example index with {len(rag.store)} labelled complaints")


if __name__ == "__main__":
    main()
