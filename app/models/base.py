from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Prediction:
    label: str
    confidence: float


class ComplaintClassifier:
    """Common interface so every approach can be evaluated the same way."""

    name = "base"

    def fit(self, texts: List[str], labels: List[str]):
        return self

    def predict(self, texts: List[str]) -> List[Prediction]:
        raise NotImplementedError

    def predict_one(self, text: str) -> Prediction:
        return self.predict([text])[0]
