"""Dataset loading.

Loads the Kaggle "University Students Complaints & Reports" CSV
(omarsobhy14) if it is placed at data/university_complaints.csv. Column
names are detected automatically, so the original file can be used as
downloaded. If the file is missing, the bundled synthetic sample is used.
"""
from __future__ import annotations

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

TEXT_CANDIDATES = ["reports", "report", "report_text", "complaint", "text", "description"]
LABEL_CANDIDATES = ["genre", "department", "category", "label", "type"]

# Maps common department labels onto the five routing categories.
LABEL_MAP = {
    "academic": "Academic", "academics": "Academic", "academic support and resources": "Academic",
    "course": "Academic", "courses": "Academic", "exams": "Academic", "teaching": "Academic",
    "finance": "Finance", "financial": "Finance", "financial support": "Finance", "fees": "Finance",
    "ict": "ICT", "it": "ICT", "technology": "ICT", "online learning": "ICT", "internet": "ICT",
    "facilities": "Facilities", "facility": "Facilities", "infrastructure": "Facilities",
    "transportation": "Facilities", "health and well-being support": "Facilities",
    "hostel": "Hostel", "housing": "Hostel", "accommodation": "Hostel", "residence": "Hostel",
}


def _find(cols, candidates):
    lower = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def normalise_label(value: str) -> str:
    key = str(value).strip().lower()
    return LABEL_MAP.get(key, str(value).strip().title())


def load_complaints(path: str | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns ``text`` and ``label`` (plus any extras)."""
    path = path or (config.COMPLAINTS_CSV if os.path.exists(config.COMPLAINTS_CSV) else config.SAMPLE_CSV)
    df = pd.read_csv(path)
    text_col = _find(df.columns, TEXT_CANDIDATES)
    label_col = _find(df.columns, LABEL_CANDIDATES)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns in {path}: {list(df.columns)}")
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    df["label"] = df["label"].map(normalise_label)
    df = df[df["text"].str.len() > 3].drop_duplicates("text").reset_index(drop=True)
    df.attrs["source"] = path
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Stratified 80:20 train/test split."""
    strat = df["label"] if df["label"].value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
