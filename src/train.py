"""Train and compare Logistic Regression, Random Forest and XGBoost.

Run:  python -m src.train
"""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLASS_NAMES, FEATURES, generate_data

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "student_data.csv"
MODEL_DIR = ROOT / "models"


def build_models(seed: int = 42) -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1)
    except ImportError:
        warnings.warn("xgboost not installed - skipping XGBoost.")
    return models


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="macro"),
        "Recall": recall_score(y_test, pred, average="macro"),
        "F1": f1_score(y_test, pred, average="macro"),
        "ROC-AUC": roc_auc_score(y_test, proba, multi_class="ovr", average="macro"),
    }


def main(seed: int = 42) -> pd.DataFrame:
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_PATH.parent.mkdir(exist_ok=True)

    df = generate_data(seed=seed)
    df.to_csv(DATA_PATH, index=False)

    X, y = df[FEATURES], df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    rows, fitted = [], {}
    for name, model in build_models(seed).items():
        cv_f1 = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="f1_macro").mean()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics["CV F1 (5-fold)"] = cv_f1
        rows.append({"Model": name, **metrics})
        fitted[name] = model
        print(f"{name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    results = pd.DataFrame(rows).set_index("Model").round(4)
    best_name = results["F1"].idxmax()
    print(f"\nBest model by macro-F1: {best_name}")

    joblib.dump(fitted, MODEL_DIR / "all_models.joblib")
    results.to_csv(MODEL_DIR / "metrics.csv")
    X_train.to_csv(MODEL_DIR / "X_train.csv", index=False)
    X_test.to_csv(MODEL_DIR / "X_test.csv", index=False)
    y_test.to_csv(MODEL_DIR / "y_test.csv", index=False)
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "best_model": best_name,
        "features": FEATURES,
        "class_names": CLASS_NAMES,
    }, indent=2))
    return results


if __name__ == "__main__":
    main()
