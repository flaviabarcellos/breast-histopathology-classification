"""Save image metrics, patient aggregation, predictions, and evaluation figures."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)

from .data import CLASS_NAMES, make_dataset


def metrics(labels, probabilities):
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "classification_report": classification_report(
            labels, predictions, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0
        )
    }


def evaluate(run):
    import pandas as pd

    config = json.loads((run / "config.json").read_text())
    frame = pd.read_csv(run / "manifest.csv", dtype={"patient_id": str})
    test = frame[frame.split == "test"].copy()
    model = tf.keras.models.load_model(run / "best.keras")
    test["malignant_probability"] = model.predict(make_dataset(test, config))[:, 1]
    test["prediction"] = (test.malignant_probability >= 0.5).astype(int)
    report = run / "evaluation"
    report.mkdir(exist_ok=True)
    test.to_csv(report / "predictions.csv", index=False)
    patients = test.groupby("patient_id").agg(
        label=("label", "first"), malignant_probability=("malignant_probability", "mean")
    )
    patients.to_csv(report / "patient_predictions.csv")
    result = {"image": metrics(test.label, test.malignant_probability),
              "patient_mean_probability": metrics(patients.label, patients.malignant_probability)}
    (report / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    ConfusionMatrixDisplay.from_predictions(test.label, test.prediction,
                                           display_labels=CLASS_NAMES)
    plt.savefig(report / "confusion_matrix.png", bbox_inches="tight")
    plt.close()
    RocCurveDisplay.from_predictions(test.label, test.malignant_probability)
    plt.savefig(report / "roc_curve.png", bbox_inches="tight")
    plt.close()
    return result
