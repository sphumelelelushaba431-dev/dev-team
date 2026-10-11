"""One command to prepare everything: data, vector database, models and evaluation."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(script, *args):
    print(f"\n=== {script} ===")
    subprocess.run([PY, str(ROOT / "scripts" / script), *args], check=True, cwd=ROOT)


if __name__ == "__main__":
    if not (ROOT / "data" / "sample_complaints.csv").exists():
        run("generate_sample_data.py")
    if not (ROOT / "data" / "university_complaints.csv").exists():
        print("\nNote: data/university_complaints.csv not found, using the synthetic sample. "
              "Download the Kaggle dataset for real results (see README).")
    run("ingest_kb.py")
    run("train.py")
    run("evaluate_complaints.py")
    run("evaluate_models.py")
    print("\nDone. Start the app with:  python run.py")
    print("Create a staff account with:  python -m flask --app app create-admin")
