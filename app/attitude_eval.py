"""Train and test RAG and LLM on the attitude dataset with stratified k-fold cross-validation.

With only 131 responses, a single 80:20 split gives a test set of 27 students,
so metrics would swing a lot between runs. Stratified 5-fold cross-validation
trains on 80% and tests on the remaining 20% five times, so every student is
tested exactly once. Metrics are reported per fold (mean and standard
deviation) and over all out-of-fold predictions.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold

from . import config
from .attitude import LABELS, load_attitude
from .models.attitude_models import LLMZeroShot, MajorityBaseline, RAGModel
from .models.llm import LLMClient

RESULTS_PATH = config.ARTIFACTS_DIR / "attitude_evaluation.json"


def _metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": float(p), "recall": float(r), "f1": float(f1)}


def evaluate(folds: int | None = None, seed: int = 42, include_llm: bool = True, progress=print):
    df = load_attitude()
    folds = folds or config.CV_FOLDS
    llm = LLMClient()
    factories = {"RAG": lambda: RAGModel(llm if llm.available else None),
                 "Majority-class baseline": MajorityBaseline}
    if include_llm and llm.available:
        factories["LLM"] = lambda: LLMZeroShot(llm)
    preds = {name: [None] * len(df) for name in factories}
    lat = {name: [] for name in factories}
    fold_scores = {name: [] for name in factories}
    details = {}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(df, df["label"]), 1):
        train, test = df.iloc[tr], df.iloc[te]
        for name, make in factories.items():
            model = make().fit(train)
            details[name] = getattr(model, "detail", model.name)
            fold_pred = []
            for idx, (_, row) in zip(te, test.iterrows()):
                t0 = time.perf_counter()
                p = model.predict_row(row).label
                lat[name].append((time.perf_counter() - t0) * 1000)
                preds[name][idx] = p
                fold_pred.append(p)
            fold_scores[name].append(_metrics(test["label"].tolist(), fold_pred))
        progress(f"fold {f}/{folds} done")

    y = df["label"].tolist()
    results = []
    for name in factories:
        overall = _metrics(y, preds[name])
        per_fold = {k: [s[k] for s in fold_scores[name]] for k in overall}
        results.append({
            "model": name, "detail": details[name], "uses_llm": name == "LLM" or (name == "RAG" and llm.available),
            **{k: round(v, 4) for k, v in overall.items()},
            "std": {k: round(float(np.std(v)), 4) for k, v in per_fold.items()},
            "per_fold": {k: [round(x, 4) for x in v] for k, v in per_fold.items()},
            "latency_ms_mean": round(float(np.mean(lat[name])), 2),
            "latency_ms_p95": round(float(np.percentile(lat[name], 95)), 2),
            "confusion_matrix": confusion_matrix(y, preds[name], labels=LABELS).tolist(),
            "per_class": {l: {k: round(v, 4) for k, v in m.items()} for l, m in
                          classification_report(y, preds[name], labels=LABELS, output_dict=True,
                                                zero_division=0).items() if l in LABELS},
        })
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"name": "AI_Chatbots_Students_Attitude_Dataset", "source": df.attrs["source"],
                    "responses": len(df), "features": "28 Likert items, degree, gender, field, comment",
                    "target": "Chatbot adoption level from Q4 (Low = never/rarely, Medium = sometimes, "
                              "High = often/very often)",
                    "class_counts": {l: int((df["label"] == l).sum()) for l in LABELS}},
        "protocol": f"Stratified {folds}-fold cross-validation (each fold trains on 80%, tests on 20%)",
        "labels": LABELS, "llm": llm.label,
        "llm_note": None if llm.available else "No LLM configured: set GROQ_API_KEY (free), run Ollama, or set "
                                               "ANTHROPIC_API_KEY / OPENAI_API_KEY, then re-run the evaluation.",
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def load_results():
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None
