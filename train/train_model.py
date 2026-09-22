"""Train the intent classifier from scratch and export it as a versioned
joblib artifact.

Pipeline: a union of word n-grams and character n-grams -> LogisticRegression.

Two decisions worth stating, both documented at length in DECISIONS.md.

FEATURES. Character n-grams alone (the obvious choice for Arabic, given its
rich morphology and the prefixes/suffixes that glue onto a stem) turned out to
over-fit the generator's templates: the model learned sentence skeletons and
missed content words, scoring 100% on its own test split while misrouting real
phrasing. Word n-grams alone miss the morphology. The union gets both: word
features carry the meaning, char features carry the robustness to spelling.

CLASSIFIER. LogisticRegression rather than LinearSVC, because the decision
policy thresholds on confidence. LinearSVC exposes only `decision_function`
margins, which have to be squashed through a softmax to fake a probability --
producing numbers that are monotonic but meaningless in absolute terms, so a
threshold like 0.45 has no defensible interpretation. LogisticRegression is
fit by maximising likelihood, so `predict_proba` is a calibrated probability
and the policy's thresholds mean what they say.

Both the vectorizer and the classifier are fit here, together, as a single
sklearn Pipeline, so the artifact is self-contained: the serving adapter never
re-implements feature extraction.

Usage (from the project root, after `python -m train.generate_dataset`):

    python -m train.train_model

Writes:
    models/intent_model.joblib   {"pipeline": ..., "version": ..., "labels": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

from train.generate_dataset import LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

MODEL_VERSION = "v1.0.0"


def load_split(name: str) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    return frame["text"].tolist(), frame["label"].tolist()


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        # Content words: what the message is actually about.
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                        # Sub-word shapes: survives the prefixes and suffixes
                        # Arabic glues onto a stem (ال-, و-, -ها, -كم), and
                        # absorbs the spelling variation of dialect writing.
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    max_iter=2000,
                    # Explicit, even though the dataset is balanced by
                    # construction: if a future label is under-represented,
                    # this keeps it from being quietly ignored.
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    x_test, y_test = load_split("test")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    print("=== validation report ===")
    print(classification_report(y_val, pipeline.predict(x_val), labels=LABELS, zero_division=0))

    print("=== test report ===")
    test_report = classification_report(
        y_test, pipeline.predict(x_test), labels=LABELS, output_dict=True, zero_division=0
    )
    print(classification_report(y_test, pipeline.predict(x_test), labels=LABELS, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "version": MODEL_VERSION,
        "labels": LABELS,
        "sklearn_version": __import__("sklearn").__version__,
    }
    artifact_path = MODELS_DIR / "intent_model.joblib"
    joblib.dump(artifact, artifact_path)

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "version": MODEL_VERSION,
                "test_accuracy": test_report["accuracy"],
                "test_macro_f1": test_report["macro avg"]["f1-score"],
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_test": len(x_test),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved {artifact_path} (version {MODEL_VERSION})")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
