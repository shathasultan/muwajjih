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
        """Return (label, calibrated_probability).

        One `predict_proba` call, with the label read off the argmax, rather
        than calling `predict` and `predict_proba` separately: two calls would
        run feature extraction twice, and -- worse -- could in principle
        disagree, handing the policy a confidence that belongs to a different
        label than the one being returned.

        The probability is a real one (see train/train_model.py on why the
        classifier is LogisticRegression), which is what makes the policy's
        thresholds interpretable rather than arbitrary.
        """
        probabilities: npt.NDArray[np.float64] = self._pipeline.predict_proba([text])[0]
        index = int(np.argmax(probabilities))
        label = str(self._pipeline.classes_[index])
        return label, float(probabilities[index])
