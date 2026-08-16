"""
Train 5 classification models on the KDD Cup 1999 network intrusion detection
dataset (10 percent subset, 494,021 rows sourced via scikit-learn), compute
evaluation metrics for each, and save models plus a comparison table.

Target: attack_category, one of {normal, dos, probe, r2l, u2r}, derived from
the original 23 fine-grained KDD Cup labels.

Models: Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes,
Random Forest.

The raw dataset is capped per class (majority classes downsampled, minority
classes kept in full) so that pipeline artifacts stay small enough for
GitHub and Streamlit Community Cloud, while preserving every instance of
the rare attack types.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_kddcup99
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "kddcup_processed.csv"
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42
PER_CLASS_CAP = 50000

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
TARGET_COL = "attack_category"

ATTACK_CATEGORY_MAP = {
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "normal": "normal",
}


def fetch_and_prepare() -> pd.DataFrame:
    bunch = fetch_kddcup99(percent10=True, as_frame=True)
    df = bunch.frame.copy()

    for col in ["protocol_type", "service", "flag", "labels"]:
        df[col] = df[col].str.decode("utf-8")

    df["labels"] = df["labels"].str.rstrip(".")
    df[TARGET_COL] = df["labels"].map(ATTACK_CATEGORY_MAP)
    df = df.drop(columns=["labels"])

    capped_parts = []
    for category, group in df.groupby(TARGET_COL):
        if len(group) > PER_CLASS_CAP:
            group = group.sample(n=PER_CLASS_CAP, random_state=RANDOM_STATE)
        capped_parts.append(group)
    capped = pd.concat(capped_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return capped


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def evaluate(y_true, y_pred, y_proba, class_labels) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=class_labels),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    df = fetch_and_prepare()
    df.to_csv(DATA_PATH, index=False)
    print(f"Prepared dataset: {df.shape}, class balance:\n{df[TARGET_COL].value_counts()}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    X = df[feature_cols]
    y = df[TARGET_COL]
    class_labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    results = {}
    for name, clf in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(numeric_cols)),
                ("model", clf),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        proba_labels = list(pipeline.classes_)
        y_proba_aligned = pd.DataFrame(y_proba, columns=proba_labels)[class_labels].values
        results[name] = evaluate(y_test, y_pred, y_proba_aligned, class_labels)

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{slug}.joblib")
        print(f"Trained {name}: {results[name]}")

    metrics_df = pd.DataFrame(results).T.round(4)
    metrics_df.index.name = "ML Model Name"
    metrics_df.to_csv(MODEL_DIR / "metrics_comparison.csv")
    print("\nComparison table:\n", metrics_df)

    with open(MODEL_DIR / "feature_schema.json", "w") as f:
        json.dump(
            {
                "numeric_cols": numeric_cols,
                "categorical_cols": CATEGORICAL_COLS,
                "target_col": TARGET_COL,
                "class_labels": class_labels,
            },
            f,
            indent=2,
        )

    test_sample = X_test.copy()
    test_sample[TARGET_COL] = y_test
    sample_size = min(3000, len(test_sample))
    sampled_parts = []
    for category, group in test_sample.groupby(TARGET_COL):
        n = max(1, round(sample_size * len(group) / len(test_sample)))
        sampled_parts.append(group.sample(n=min(n, len(group)), random_state=RANDOM_STATE))
    test_sample = pd.concat(sampled_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_sample.to_csv(TEST_DATA_PATH, index=False)
    print(f"\nSaved {len(test_sample)}-row test_data.csv to {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
