import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Network Intrusion Detection Classifier", layout="wide")


@st.cache_resource
def load_models():
    models = {}
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        name = path.stem.replace("_", " ").title().replace("Knn", "kNN")
        models[name] = joblib.load(path)
    return models


@st.cache_data
def load_schema():
    with open(MODEL_DIR / "feature_schema.json") as f:
        return json.load(f)


@st.cache_data
def load_comparison_table():
    return pd.read_csv(MODEL_DIR / "metrics_comparison.csv", index_col=0)


def compute_metrics(y_true, y_pred, y_proba, class_labels):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=class_labels),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("Network Intrusion Detection, Classification Demo")
    st.caption(
        "Upload test data from the KDD Cup 1999 dataset, pick a model, and review its "
        "predictions against the 6 evaluation metrics required by the assignment. "
        "The target has 5 classes: normal, dos, probe, r2l, and u2r."
    )

    schema = load_schema()
    models = load_models()
    target_col = schema["target_col"]
    class_labels = schema["class_labels"]

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
        model_name = st.selectbox("Select a model", list(models.keys()))
        st.markdown("---")
        st.subheader("All Models, Validation Set Accuracies")
        st.caption("Computed during training on a held-out validation split.")
        st.dataframe(load_comparison_table(), use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a CSV file using the sidebar to get started. A sample `test_data.csv` is included in the repository.")
        return

    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    feature_cols = schema["numeric_cols"] + schema["categorical_cols"]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded file is missing required columns: {missing_cols}")
        return

    pipeline = models[model_name]
    X = df[feature_cols]
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    proba_labels = list(pipeline.classes_)
    y_proba_aligned = pd.DataFrame(y_proba, columns=proba_labels)[class_labels].values

    result_df = df.copy()
    result_df["Predicted"] = y_pred
    st.subheader(f"Predictions, {model_name}")
    st.dataframe(result_df.head(20), use_container_width=True)

    has_ground_truth = target_col in df.columns
    if has_ground_truth:
        y_true = df[target_col]

        st.subheader("Evaluation Metrics on Uploaded Data")
        metrics = compute_metrics(y_true, y_pred, y_proba_aligned, class_labels)
        cols = st.columns(len(metrics))
        for col, (metric_name, value) in zip(cols, metrics.items()):
            col.metric(metric_name, f"{value:.4f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=class_labels)
            fig, ax = plt.subplots(figsize=(5, 4.2))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with right:
            st.subheader("Classification Report")
            report = classification_report(
                y_true, y_pred, labels=class_labels, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    else:
        st.warning(
            f"Uploaded file has no '{target_col}' column, showing predictions only. "
            "Include the true label column to see evaluation metrics and the confusion matrix."
        )


if __name__ == "__main__":
    main()
