"""Evaluates every approach with accuracy, precision, recall, F1 and latency."""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from . import config


def evaluate_model(model, texts, labels, label_set):
    preds, latencies = [], []
    for t in texts:
        t0 = time.perf_counter()
        preds.append(model.predict_one(t).label)
        latencies.append((time.perf_counter() - t0) * 1000)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, labels=label_set, average="macro", zero_division=0)
    return {
        "model": model.name, "n": len(texts),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4),
        "latency_ms_mean": round(float(np.mean(latencies)), 2),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "unknown_rate": round(preds.count("Unknown") / len(preds), 4),
        "confusion_matrix": {"labels": label_set,
                             "matrix": confusion_matrix(labels, preds, labels=label_set).tolist()},
        "per_class": classification_report(labels, preds, labels=label_set, output_dict=True, zero_division=0),
    }


RAG_RESULTS_PATH = config.ARTIFACTS_DIR / "rag_evaluation.json"


def evaluate_rag(progress=print):
    """RAG on student complaints for the staff dashboard: build the retrieval index from the 80%
    training split, then test RAG and the intent-matching baseline on the held-out 20%."""
    from .data import load_complaints, split
    from .models.intent_baseline import IntentMatcher
    from .models.rag_classifier import RAGClassifier

    df = load_complaints()
    train, test = split(df)
    texts, labels = test["text"].tolist(), test["label"].tolist()
    label_set = sorted(df["label"].unique())
    progress("Building the RAG index")
    rag = RAGClassifier().fit(train["text"].tolist(), train["label"].tolist())
    models = [("RAG", f"{rag.store.backend} retrieval over {len(train)} labelled reports "
                      f"({rag.embedder.kind} embeddings, k={rag.k}), similarity vote", rag),
              ("Intent-matching baseline", "keyword rules", IntentMatcher())]
    results = []
    for name, detail, model in models:
        progress(f"Testing {name}")
        r = evaluate_model(model, texts, labels, label_set)
        results.append({**r, "model": name, "detail": detail, "confusion_matrix": r["confusion_matrix"]["matrix"]})
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"source": df.attrs["source"], "synthetic": df.attrs["source"] == config.SAMPLE_CSV,
                    "complaints": len(df), "train": len(train), "test": len(test),
                    "class_counts": {l: int((df["label"] == l).sum()) for l in label_set}},
        "protocol": "Stratified 80:20 train/test split",
        "labels": label_set, "results": results,
    }
    RAG_RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_rag_results():
    return json.loads(RAG_RESULTS_PATH.read_text()) if RAG_RESULTS_PATH.exists() else None


def write_report(results, dataset_info, path_json=None, path_md=None):
    path_json = path_json or config.ARTIFACTS_DIR / "evaluation.json"
    path_md = path_md or config.ARTIFACTS_DIR / "evaluation_report.md"
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "dataset": dataset_info, "results": results}
    path_json.write_text(json.dumps(payload, indent=2))
    lines = [
        "# CampusVoice model evaluation", "",
        f"Dataset: `{dataset_info['source']}` ({dataset_info['train']} train / {dataset_info['test']} test, "
        f"stratified 80:20 split)", "",
        "| Model | Accuracy | Precision | Recall | F1-score | Mean latency (ms) | p95 latency (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {r['model']} | skipped: {r['skipped']} | | | | | |")
            continue
        lines.append(f"| {r['model']} | {r['accuracy']:.1%} | {r['precision']:.1%} | {r['recall']:.1%} | "
                     f"{r['f1']:.1%} | {r['latency_ms_mean']} | {r['latency_ms_p95']} |")
    lines += ["", "Precision, recall and F1 are macro-averaged across departments. Latency is measured per "
              "complaint on the machine that ran the evaluation."]
    path_md.write_text("\n".join(lines))
    return path_json, path_md
