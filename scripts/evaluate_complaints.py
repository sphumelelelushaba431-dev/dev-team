"""Compare the four approaches from the paper on the 20% test split.

  python scripts/evaluate_complaints.py              # all models; LLM-based ones use up to 100 test items
  python scripts/evaluate_complaints.py --llm-limit 0  # evaluate LLM-based models on the whole test set
"""
import argparse

import _path  # noqa: F401
from app import config
from app.data import load_complaints, split
from app.evaluation import RAG_RESULTS_PATH, evaluate_model, evaluate_rag, write_report
from app.models.deep_classifier import DeepClassifier
from app.models.intent_baseline import IntentMatcher
from app.models.llm import LLMClient
from app.models.llm_classifier import LLMClassifier
from app.models.rag_classifier import RAGClassifier
from app.rag.pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-limit", type=int, default=100, help="max test items for LLM-based models (0 = all)")
    args = ap.parse_args()

    df = load_complaints()
    train, test = split(df)
    texts, labels = test["text"].tolist(), test["label"].tolist()
    label_set = sorted(df["label"].unique())
    llm = LLMClient()
    limit = len(texts) if args.llm_limit in (0, None) else min(args.llm_limit, len(texts))

    results = []
    print(f"Evaluating on {len(texts)} test complaints ({df.attrs['source']})")

    print("- Chatbot intent matching")
    results.append(evaluate_model(IntentMatcher(), texts, labels, label_set))

    print("- Deep learning")
    deep = DeepClassifier.load(config.ARTIFACTS_DIR / "deep_classifier")
    results.append(evaluate_model(deep, texts, labels, label_set))

    print("- Large language model")
    if llm.available:
        results.append(evaluate_model(LLMClassifier(llm, label_set), texts[:limit], labels[:limit], label_set))
    else:
        results.append({"model": "LLM zero-shot", "skipped": "no LLM configured (set ANTHROPIC_API_KEY, "
                                                              "OPENAI_API_KEY or install transformers)"})

    print("- Retrieval-augmented generation")
    kb = RAGPipeline(llm=llm)
    rag = RAGClassifier(llm=llm, kb_pipeline=kb).load(config.ARTIFACTS_DIR / "rag_examples")
    n = limit if llm.available else len(texts)
    results.append(evaluate_model(rag, texts[:n], labels[:n], label_set))
    if llm.available:  # also report retrieval-only RAG for comparison
        results.append(evaluate_model(RAGClassifier().load(config.ARTIFACTS_DIR / "rag_examples"),
                                      texts, labels, label_set))

    info = {"source": df.attrs["source"], "train": len(train), "test": len(test), "labels": label_set}
    j, m = write_report(results, info)
    print()
    print(m.read_text())
    print(f"\nSaved {j} and {m}")
    evaluate_rag(progress=lambda msg: None)  # results for the staff Model performance dashboard
    print(f"Saved {RAG_RESULTS_PATH}")


if __name__ == "__main__":
    main()
