"""The ONLY file in this service allowed to import sklearn/joblib.

If the team ever swaps the model family (a transformer, a hosted LLM), they
add a sibling adapter and change one line in the composition root -- the
service and API layers never notice.
"""

from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.pipeline import Pipeline


class SklearnIntentModel:
    """Adapter wrapping a fitted sklearn Pipeline behind the `IntentModel`
    port. The pipeline itself owns feature extraction (TF-IDF) and
    classification, so this adapter never re-implements preprocessing --
    reusing exactly what train/train_model.py fit avoids train/serve skew."""

    def __init__(self, pipeline: Pipeline, model_version: str, labels: list[str]) -> None:
        self._pipeline = pipeline
        self.model_version = model_version
        self.labels = labels

    @classmethod
    def load(cls, path: str | Path) -> "SklearnIntentModel":
        """Explicit, failable load -- called ONLY from a composition root.

        Never happens at import time: importing this module costs nothing
        and requires no artifact on disk, which is exactly what keeps unit
        tests fast (see tests/unit/test_classifier.py).
        """
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Model artifact not found: {resolved.resolve()}")

        artifact = joblib.load(resolved)
        pipeline = artifact["pipeline"]
        labels = artifact["labels"]
        version = artifact.get("version", "unknown")

        # A pipeline whose classes don't match the domain's label set would
        # silently mis-map predictions -- fail at load time instead.
        trained_classes = sorted(pipeline.classes_)
        if trained_classes != sorted(labels):
            raise ValueError(
                f"Artifact label mismatch: pipeline.classes_={trained_classes} "
                f"but artifact['labels']={sorted(labels)}"
            )

        return cls(pipeline, version, labels)

    def predict(self, text: str) -> tuple[str, float]:
        label = str(self._pipeline.predict([text])[0])

        # LinearSVC has no predict_proba; decision_function's margins are
        # converted to a pseudo-confidence via softmax so the API always has
        # a bounded [0, 1] number, without silently pretending the model is
        # calibrated probability output.
        scores = self._pipeline.decision_function([text])[0]
        confidence = _softmax_max(scores)
        return label, confidence


def _softmax_max(scores: npt.NDArray[np.float64]) -> float:
    exp = np.exp(scores - np.max(scores))
    return float(np.max(exp / exp.sum()))
