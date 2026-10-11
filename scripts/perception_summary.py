"""Summarise student attitudes in AI_Chatbots_Students_Attitude_Dataset (paper section 3.2).

    python scripts/perception_summary.py

Prints, for each Likert item, the mean score (1-5) and the share of students who
agree or strongly agree, plus how often students use chatbots. This supports the
paper's rationale that students are receptive to chatbots.
"""
import _path  # noqa: F401
import pandas as pd
from app.attitude import likert_cols, load_attitude

df = load_attitude()
cols = likert_cols(df)
summary = pd.DataFrame({"mean": df[cols].mean().round(2),
                        "agree_%": (df[cols].ge(4).mean() * 100).round(1)}).sort_values("agree_%", ascending=False)
print(f"{len(df)} responses, {len(cols)} Likert items\n")
print("How often students use AI chatbots (Q4):")
print(df["Q4"].value_counts().to_string(), "\n")
print("Adoption level used as the prediction target:")
print(df["label"].value_counts().to_string(), "\n")
print(summary.to_string())
out = df.attrs["source"].rsplit(".", 1)[0] + "_summary.csv"
summary.to_csv(out)
print(f"\nSaved {out}")
