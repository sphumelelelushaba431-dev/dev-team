"""Evaluate the paper's RAG classifier (+ baselines) on the Kaggle complaints dataset.

Protocol (README: Evaluation protocol):
  * Stratified 80:20 train/test split (random_state=42). The test split is
    never seen while the retrieval index is built.
  * RAGClassifier (app/models/rag_classifier.py): the 80% training complaints
    are embedded and stored in the FAISS vector database; for each unseen test
    complaint the retriever finds the k most similar training complaints and:
      - with an LLM: wraps them (plus matching policy chunks) into the prompt as
        grounded context and the LLM decides the category;
      - without an LLM: a similarity-weighted vote (power = vote_power).
  * Baseline: IntentMatcher, the rule-based chatbot intent matching the paper
    compares RAG against.

Examples:
  python scripts/evaluate_kaggle_rag.py                    # all models, all 200 test items via LLM
  python scripts/evaluate_kaggle_rag.py --llm-limit 0      # all test items (0 = everything)
  python scripts/evaluate_kaggle_rag.py --llm-limit 30     # cap LLM calls for a quick run
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import _path  # noqa: F401

from app import config
from app.data import load_complaints, split
from app.evaluation import evaluate_model
from app.models.intent_baseline import IntentMatcher
from app.models.llm import LLMClient
from app.models.rag_classifier import RAGClassifier


def fmt_row(name, r):
    if "skipped" in r:
        return f"| {name} | skipped: {r['skipped']} | | | | | |"
    return (f"| {name} | {r['accuracy']:.1%} | {r['precision']:.1%} | {r['recall']:.1%} | "
            f"{r['f1']:.1%} | {r['latency_ms_mean']:.1f} | {r['latency_ms_p95']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="number of retrieved neighbours (CV-tuned on train only)")
    ap.add_argument("--vote-power", type=float, default=4.0, help="vote sharpness (CV-tuned on train only)")
    ap.add_argument("--llm-limit", type=int, default=0, help="max test items for the LLM-grounded RAG (0 = all)")
    args = ap.parse_args()

    df = load_complaints()
    train, test = split(df)  # stratified 80:20, random_state=42
    texts, labels = test["text"].tolist(), test["label"].tolist()
    label_set = sorted(df["label"].unique())
    limit = len(texts) if args.llm_limit in (0, None) else min(args.llm_limit, len(texts))

    print(f"Dataset        : {df.attrs['source']}")
    print(f"Complaints     : {len(df)} total -> {len(train)} train (80%) / {len(test)} test (20%)")
    print(f"Categories     : {len(label_set)} -> {label_set}")

    llm = LLMClient()
    print(f"LLM            : {llm.label if llm.available else 'not configured - RAG falls back to retrieval vote'}")

    print("Building the retrieval index from the 80% training split ...")
    t0 = time.perf_counter()
    rag = RAGClassifier(k=args.k, vote_power=args.vote_power).fit(train["text"].tolist(), train["label"].tolist())
    print(f"Index built    : {len(rag.store)} vectors ({rag.store.backend}/{rag.embedder.kind}) "
          f"in {time.perf_counter() - t0:.1f}s")

    results = []

    print(f"\nTesting on {len(test)} unseen complaints ...")
    print("- Chatbot intent matching (baseline, all test items)")
    results.append(evaluate_model(IntentMatcher(), texts, labels, label_set))

    print("- RAG, retrieval only (k-NN vote, all test items)")
    rag.name = f"RAG retrieval-only ({rag.store.backend}/{rag.embedder.kind}, k={args.k})"
    results.append(evaluate_model(rag, texts, labels, label_set))

    if llm.available:
        print(f"- RAG, retrieval + LLM grounding ({llm.label}, {limit} test items)")
        rag_llm = RAGClassifier(llm=llm, k=args.k, vote_power=args.vote_power)
        rag_llm.store, rag_llm.embedder, rag_llm.labels = rag.store, rag.embedder, rag.labels
        results.append(evaluate_model(rag_llm, texts[:limit], labels[:limit], label_set))
    else:
        results.append({"model": "RAG retrieval + LLM", "skipped": "no LLM available (start Ollama / set a key)"})

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "source": df.attrs["source"], "complaints": len(df),
            "train": len(train), "test": len(test),
            "split": "stratified 80:20, random_state=42",
            "class_counts": {l: int((df["label"] == l).sum()) for l in label_set},
        },
        "models": {"k": args.k, "vote_power": args.vote_power, "backend": rag.store.backend,
                   "embedder": rag.embedder.kind, "llm": llm.label if llm.available else None,
                   "llm_test_items": limit},
        "results": results,
    }
    path = config.ARTIFACTS_DIR / "kaggle_rag_evaluation.json"
    path.write_text(json.dumps(payload, indent=2))

    print()
    print("## Summary")
    print("| Model | Accuracy | Precision | Recall | F1 | Mean ms | p95 ms |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(fmt_row(r["model"], r))

    best = max((r for r in results if "skipped" not in r), key=lambda r: r["f1"])
    print(f"\nBest model by macro F1: {best['model']} ({best['f1']:.1%})")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()