"""Train five classifiers on the UCI Dry Bean dataset and save them to model/.

Run:  python train_models.py
Outputs:
  model/*.joblib        - one fitted pipeline per classifier
  model/metrics.json    - test-set metrics for every model
  test_data.csv         - held-out 20% test split (features + Class)
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from metrics_utils import FEATURE_COLS, MODEL_FILES, TARGET_COL, evaluate

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dry_bean_full.csv"
MODEL_DIR = ROOT / "model"
SEED = 42

# scikit-learn 1.9 marks SVC(probability=True) as deprecated (removal in 1.11).
# It still works in the pinned version, so the warning is silenced here.
warnings.filterwarnings("ignore", message=".*probability.*", category=FutureWarning)


def build_models():
    """Every model is a Pipeline, so the scaler is saved together with it."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(max_depth=12, min_samples_leaf=3,
                                           random_state=SEED)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=9, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                        random_state=SEED)),
        ]),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {df.shape[0]} rows")
    print("\nClass distribution:\n", df[TARGET_COL].value_counts().to_string())

    X, y = df[FEATURE_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name])
        print(f"Trained and saved: {name}")

    table = pd.DataFrame(results).T.round(4)
    table.index.name = "ML Model Name"
    print("\n=== Test-set metrics (macro-averaged) ===")
    print(table.to_string())

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({
            "sklearn_version": sklearn.__version__,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "metrics": table.to_dict(orient="index"),
        }, f, indent=2)
    print(f"\nscikit-learn version used: {sklearn.__version__}")


if __name__ == "__main__":
    main()
