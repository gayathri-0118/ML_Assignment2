"""Shared evaluation helpers used by both train_models.py and app.py."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COL = "Class"

FEATURE_COLS = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def multiclass_auc(y_true, proba, classes):
    """Macro one-vs-rest AUC over the classes that are present in y_true.

    Equals sklearn's roc_auc_score(..., multi_class="ovr", average="macro")
    when every class is present, and still works when an uploaded file
    contains only a subset of the classes.
    """
    y_true = np.asarray(y_true)
    scores = []
    for idx, cls in enumerate(classes):
        positives = y_true == cls
        if positives.any() and (~positives).any():
            scores.append(roc_auc_score(positives.astype(int), proba[:, idx]))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(model, X, y):
    """Return the six assignment metrics for a fitted classifier."""
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": multiclass_auc(y, proba, model.classes_),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
