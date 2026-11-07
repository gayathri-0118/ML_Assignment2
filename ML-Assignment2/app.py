"""Streamlit app: evaluate pre-trained Dry Bean classifiers on an uploaded test CSV.

The models in model/ are only loaded and used for prediction here.
Nothing is retrained on the uploaded data.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import FEATURE_COLS, MODEL_FILES, TARGET_COL, evaluate

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_TEST = ROOT / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_training_metrics():
    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def validate(df):
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        return f"The uploaded file is missing these columns: {', '.join(missing)}"
    if df[FEATURE_COLS].isna().any().any():
        return "Some feature values are empty. Please upload a file without missing values."
    return None


def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGn", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted bean type")
    ax.set_ylabel("Actual bean type")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return fig


models = load_models()

# ---------------- Sidebar ----------------
st.sidebar.title("🫘 Controls")
model_name = st.sidebar.selectbox("Choose a classifier", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"])
if SAMPLE_TEST.exists():
    st.sidebar.download_button(
        "Download sample test_data.csv",
        data=SAMPLE_TEST.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )
st.sidebar.caption(
    f"The CSV needs the 16 feature columns and the `{TARGET_COL}` column. "
    "Models are pre-trained; uploaded data is used only for evaluation."
)

# ---------------- Header ----------------
st.title("Dry Bean Variety Classifier Lab")
st.write(
    "Seven varieties of dry beans (UCI Dry Bean dataset) are classified from 16 "
    "shape features extracted from grain images. Upload the held-out test file "
    "to see how each of the five trained models performs."
)

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. "
            "You can download the sample test file there as well.")
    train_info = load_training_metrics()
    if train_info:
        st.subheader("Metrics recorded at training time (20% hold-out split)")
        st.dataframe(pd.DataFrame(train_info["metrics"]).T.style.format("{:.4f}")
                     .highlight_max(axis=0, color="#c7e9c0"),
                     width="stretch")
    st.stop()

data = pd.read_csv(uploaded)
error = validate(data)
if error:
    st.error(error)
    st.stop()

X, y = data[FEATURE_COLS], data[TARGET_COL].astype(str)
unknown = sorted(set(y) - set(models[model_name].classes_))
if unknown:
    st.error(f"Unknown class labels in `{TARGET_COL}`: {', '.join(unknown)}")
    st.stop()

st.success(f"Loaded **{len(data)}** rows and **{len(data.columns)}** columns from "
           f"`{uploaded.name}`.")

tab_single, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏁 Compare all models", "🔍 Data & predictions"]
)

# ---------------- Selected model ----------------
with tab_single:
    model = models[model_name]
    scores = evaluate(model, X, y)
    y_pred = model.predict(X)

    st.subheader(f"Evaluation metrics: {model_name}")
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged over the classes. "
               "AUC is the macro one-vs-rest ROC AUC.")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y, y_pred, list(model.classes_)))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.style.format("{:.3f}"), width="stretch")

# ---------------- Compare all ----------------
with tab_compare:
    st.subheader("All five models on the uploaded data")
    rows = {name: evaluate(m, X, y) for name, m in models.items()}
    compare_df = pd.DataFrame(rows).T
    compare_df.index.name = "ML Model Name"
    st.dataframe(compare_df.style.format("{:.4f}").highlight_max(axis=0, color="#c7e9c0"),
                 width="stretch")
    best = compare_df["MCC"].idxmax()
    st.markdown(f"Best model by MCC on this file: **{best}** "
                f"(MCC = {compare_df.loc[best, 'MCC']:.4f})")

    chosen = st.multiselect("Metrics to plot", list(compare_df.columns),
                            default=["Accuracy", "F1", "MCC"])
    if chosen:
        fig, ax = plt.subplots(figsize=(9, 4))
        compare_df[chosen].plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black")
        ax.set_ylim(max(0.0, compare_df[chosen].min().min() - 0.05), 1.0)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        plt.xticks(rotation=20, ha="right")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)

# ---------------- Data & predictions ----------------
with tab_data:
    out = data.copy()
    out["Predicted"] = y_pred
    out["Correct"] = out["Predicted"] == y
    st.markdown(f"Predictions from **{model_name}** "
                f"({out['Correct'].sum()} of {len(out)} correct)")
    show_wrong = st.checkbox("Show only misclassified rows")
    st.dataframe(out[~out["Correct"]] if show_wrong else out, width="stretch")
    st.download_button("Download predictions as CSV",
                       out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")
