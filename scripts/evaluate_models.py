"""Train and test RAG and LLM on AI_Chatbots_Students_Attitude_Dataset.

    python scripts/evaluate_models.py            # 5-fold cross-validation
    python scripts/evaluate_models.py --folds 10

Results are saved to artifacts/attitude_evaluation.json and shown on the staff
Model performance dashboard. Set an LLM provider in .env to include the LLM.
"""
import argparse

import _path  # noqa: F401
from app.attitude_eval import RESULTS_PATH, evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None)
    args = ap.parse_args()
    payload = evaluate(folds=args.folds)
    print(f"\n{payload['dataset']['name']}: {payload['dataset']['responses']} responses, {payload['protocol']}")
    print(f"{'Model':<26}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Latency ms':>12}")
    for r in payload["results"]:
        print(f"{r['model']:<26}{r['accuracy']:>10.1%}{r['precision']:>11.1%}{r['recall']:>9.1%}{r['f1']:>8.1%}"
              f"{r['latency_ms_mean']:>12}")
    if payload["llm_note"]:
        print("\nNote:", payload["llm_note"])
    print(f"\nSaved {RESULTS_PATH}")
