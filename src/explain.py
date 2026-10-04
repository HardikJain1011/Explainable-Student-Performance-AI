"""SHAP and LIME helpers that work for all three model types."""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.pipeline import Pipeline


def _split_pipeline(model):
    """Return (estimator, transform_fn). Handles the scaled Logistic Regression."""
    if isinstance(model, Pipeline):
        scaler, est = model[:-1], model[-1]
        return est, (lambda X: scaler.transform(X))
    return model, (lambda X: np.asarray(X))


def _to_3d(sv):
    """Normalise SHAP output to (n_samples, n_features, n_classes) across versions."""
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    sv = np.asarray(sv)
    return sv[:, :, None] if sv.ndim == 2 else sv


def build_shap_explainer(model, X_background: pd.DataFrame):
    est, tf = _split_pipeline(model)
    if est.__class__.__name__ == "LogisticRegression":
        return shap.LinearExplainer(est, tf(X_background)), est, tf
    return shap.TreeExplainer(est), est, tf


def shap_values(bundle, X: pd.DataFrame):
    """-> array (n_samples, n_features, n_classes)."""
    explainer, _, tf = bundle
    return _to_3d(explainer.shap_values(tf(X)))


def contribution_table(sv_row: np.ndarray, x_row: pd.Series) -> pd.DataFrame:
    """One class' SHAP values for one student -> table sorted by |impact|."""
    df = pd.DataFrame({"feature": x_row.index, "value": x_row.values, "shap": sv_row})
    return df.sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)


def build_lime_explainer(X_train: pd.DataFrame, class_names):
    return LimeTabularExplainer(
        X_train.values,
        feature_names=list(X_train.columns),
        class_names=list(class_names),
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def lime_explain(lime_explainer, model, x_row: pd.Series, class_idx: int,
                 num_features: int = 7):
    cols = list(x_row.index)

    def predict_fn(arr):
        return model.predict_proba(pd.DataFrame(arr, columns=cols))

    exp = lime_explainer.explain_instance(
        x_row.values, predict_fn, labels=[class_idx],
        num_features=num_features, num_samples=3000)
    return exp.as_list(label=class_idx)
