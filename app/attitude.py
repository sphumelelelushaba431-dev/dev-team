"""AI_Chatbots_Students_Attitude_Dataset: loading, cleaning and features.

The survey has 131 student responses:
  Q1  degree (Bachelor / Master)          Q2  field of study      Q3  gender
  Q4  how often the student uses AI chatbots (Never ... Very often)
  Q5.1-Q9.5  28 Likert items on chatbot use, usefulness and trust
  Q10 optional free-text comment

Prediction task used to train and test the RAG and LLM models
-------------------------------------------------------------
Predict each student's chatbot ADOPTION LEVEL from their attitudes:
  Low    = Never or Rarely uses chatbots
  Medium = Sometimes
  High   = Often or Very often
The target (Q4) is never used as an input feature, so there is no leakage.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

DATASET_PATH = config.DATA_DIR / "AI_Chatbots_Students_Attitude_Dataset_EN.xlsx"
LIKERT = {"strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5}
LIKERT_TEXT = {v: k.title() for k, v in LIKERT.items()}
USAGE_TO_LEVEL = {"never": "Low", "rarely": "Low", "sometimes": "Medium", "often": "High", "very often": "High"}
LABELS = ["Low", "Medium", "High"]
LABEL_DESCRIPTIONS = {
    "Low": "never or rarely uses AI chatbots",
    "Medium": "sometimes uses AI chatbots",
    "High": "often or very often uses AI chatbots",
}
# The survey groups items into blocks; the original wording is not shipped with the
# dataset, so blocks are described by their numbering.
BLOCKS = {"Q5": "block 5", "Q6": "block 6", "Q7": "block 7", "Q8": "block 8", "Q9": "block 9"}


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    # fix Cyrillic look-alike characters left by the original encoding (e.g. 'Îften', 'Òhey')
    text = text.replace("Î", "O").replace("Ò", "T").replace("Ñ", "C").replace("Å", "E")
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text in {".", "-", "no", "No", "none", "None"} else text


def load_attitude(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or DATASET_PATH)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    likert_cols = [c for c in df.columns if re.fullmatch(r"Q[5-9]\.\d+", c)]
    for c in ["Q1", "Q2", "Q3", "Q4", "Q10"]:
        df[c] = df[c].map(_clean_text)
    for c in likert_cols:
        df[c] = df[c].map(lambda v: LIKERT.get(_clean_text(v).lower(), 3))
    df["label"] = df["Q4"].str.lower().map(USAGE_TO_LEVEL)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df.attrs["likert_cols"] = likert_cols
    df.attrs["source"] = str(path)
    return df


def likert_cols(df: pd.DataFrame):
    return df.attrs.get("likert_cols") or [c for c in df.columns if re.fullmatch(r"Q[5-9]\.\d+", c)]


def profile_text(row: pd.Series, cols) -> str:
    """Readable description of one respondent (used in LLM prompts). Q4 is excluded."""
    parts = [f"Degree: {row['Q1'] or 'unknown'}. Field: {row['Q2'] or 'unknown'}. Gender: {row['Q3'] or 'unknown'}."]
    for block, desc in BLOCKS.items():
        items = [c for c in cols if c.startswith(block + ".")]
        if items:
            answers = ", ".join(f"{c}={LIKERT_TEXT[int(row[c])]}" for c in items)
            mean = np.mean([row[c] for c in items])
            parts.append(f"Survey {desc} (mean {mean:.1f}/5): {answers}.")
    if row.get("Q10"):
        parts.append(f"Comment: \"{row['Q10'][:400]}\"")
    return " ".join(parts)


def feature_matrix(df: pd.DataFrame, cols=None) -> np.ndarray:
    """Numeric vectors for retrieval: centred Likert answers plus degree and gender."""
    cols = cols or likert_cols(df)
    X = df[cols].to_numpy(dtype="float32") - 3.0          # centre on Neutral
    extra = np.column_stack([
        (df["Q1"].str.lower() == "master").astype("float32"),
        (df["Q3"].str.lower() == "male").astype("float32"),
    ])
    return np.hstack([X, extra]).astype("float32")


def row_from_answers(answers: dict, cols) -> pd.Series:
    """Build a respondent row from an API payload ({'Q5.1': 'Agree', ..., 'Q10': '...'})."""
    data = {"Q1": answers.get("Q1", ""), "Q2": answers.get("Q2", ""), "Q3": answers.get("Q3", ""),
            "Q10": _clean_text(answers.get("Q10", ""))}
    for c in cols:
        v = answers.get(c, 3)
        data[c] = int(v) if str(v).isdigit() else LIKERT.get(str(v).strip().lower(), 3)
    return pd.Series(data)
