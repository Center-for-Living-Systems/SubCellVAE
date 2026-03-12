"""
Functions for training classifiers on latent features and evaluating them.

All plot functions both display (plt.show) and optionally save to disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_df(df_or_path) -> pd.DataFrame:
    if isinstance(df_or_path, (str, Path)):
        return pd.read_csv(df_or_path)
    return df_or_path


def _save_fig(fig: plt.Figure, save_path: Optional[str | Path], dpi: int = 200):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


# --------------------------------------------------------------------------- #
#  Data preparation
# --------------------------------------------------------------------------- #

def prepare_classification_data(
    label_df,
    latent_df,
    target_col: str,
    label_order: Sequence[str],
    latent_cols: Optional[list[str]] = None,
    *,
    merge_on: str = "unique_ID",
    exclude_labels: Optional[Sequence[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Merge label and latent spreadsheets, encode the target, and split train/val.

    Parameters
    ----------
    label_df : DataFrame or path to CSV
        Contains *target_col* and the merge key.
    latent_df : DataFrame or path to CSV
        Contains latent feature columns and the merge key.
    target_col : str
        Column with categorical labels.
    label_order : sequence of str
        Ordered class names; indices become encoded IDs.
    latent_cols : list of str, optional
        Feature columns in *latent_df*. Auto-detected as ``lat_d*`` if not given.
    merge_on : str
        Column to merge the two DataFrames on.
    exclude_labels : sequence of str, optional
        Labels to drop before training (e.g. ``["Uncertain"]``).
    test_size, random_state : split parameters.

    Returns
    -------
    dict with keys:
        ``df_merged``, ``df_clean``,
        ``X_train``, ``X_val``, ``y_train``, ``y_val``,
        ``train_idx``, ``val_idx``, ``latent_cols``, ``label_to_id``.
    """
    label_df = _load_df(label_df)
    latent_df = _load_df(latent_df)

    if latent_cols is None:
        latent_cols = sorted(
            [c for c in latent_df.columns if c.startswith("lat_d")],
            key=lambda c: int(c.split("lat_d")[1]),
        )

    merged = latent_df.merge(
        label_df[[merge_on, target_col]].drop_duplicates(),
        on=merge_on,
        how="inner",
    )

    if exclude_labels:
        merged = merged[~merged[target_col].isin(exclude_labels)].copy()

    label_to_id = {label: i for i, label in enumerate(label_order)}
    merged["_target_enc"] = merged[target_col].map(label_to_id)
    merged = merged.dropna(subset=["_target_enc"])
    merged["_target_enc"] = merged["_target_enc"].astype(int)

    X = merged[latent_cols].values
    y = merged["_target_enc"].values

    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return {
        "df_merged": merged,
        "X_train": X[train_idx],
        "X_val": X[val_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "latent_cols": latent_cols,
        "label_to_id": label_to_id,
    }


# --------------------------------------------------------------------------- #
#  Training
# --------------------------------------------------------------------------- #

_CLASSIFIERS = {
    "logistic_regression": lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ]),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "gradient_boosting",
    *,
    sample_weight: Optional[np.ndarray] = None,
    save_path: Optional[str | Path] = None,
):
    """
    Train a classifier and optionally save it.

    Parameters
    ----------
    X_train, y_train : arrays
    method : str
        One of ``"logistic_regression"``, ``"random_forest"``, ``"gradient_boosting"``.
    sample_weight : array, optional
    save_path : str or Path, optional
        If given, ``joblib.dump`` the trained model here.

    Returns
    -------
    Trained classifier object.
    """
    if method not in _CLASSIFIERS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_CLASSIFIERS)}")

    clf = _CLASSIFIERS[method]()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    clf.fit(X_train, y_train, **fit_kwargs)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)

    return clf


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #

def evaluate_classifier(
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """
    Predict on a validation set and return metrics.

    Returns
    -------
    dict with ``y_pred``, ``accuracy``, ``balanced_accuracy``,
    ``f1_macro``, ``f1_weighted``, ``report`` (str), ``confusion_matrix``.
    """
    y_pred = clf.predict(X_val)
    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_val, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_val, y_pred),
        "f1_macro": f1_score(y_val, y_pred, average="macro"),
        "f1_weighted": f1_score(y_val, y_pred, average="weighted"),
        "report": classification_report(y_val, y_pred, target_names=class_names),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
    }


# --------------------------------------------------------------------------- #
#  Confusion-matrix plotting
# --------------------------------------------------------------------------- #

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    *,
    normalize: bool = False,
    method_str: str = "",
    cmap: str = "Blues",
    figsize: tuple[float, float] = (6, 6),
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Display (and optionally save) a confusion-matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : 1-D int arrays
    class_names : sequence of str
    normalize : bool
        If True, rows are normalised to sum to 1.
    method_str : str
        Label for the title (e.g. ``"Gradient Boosting"``).
    """
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    if normalize:
        cm_show = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    else:
        cm_show = cm

    n = min(len(class_names), cm.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.sqrt(cm_show[:n, :n]).max() * 0.8 if not normalize else None
    ax.imshow(
        np.sqrt(cm_show[:n, :n]) if not normalize else cm_show[:n, :n],
        cmap=cmap,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names[:n], rotation=45, ha="right")
    ax.set_yticklabels(class_names[:n])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    norm_tag = " (Normalized)" if normalize else " (Counts)"
    ax.set_title(f"{method_str} Confusion Matrix{norm_tag}, ACC={acc:.2f}")

    for i in range(n):
        for j in range(n):
            txt = f"{cm_show[i, j]:.2f}" if normalize else str(cm[i, j])
            ax.text(j, i, txt, ha="center", va="center")

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# --------------------------------------------------------------------------- #
#  Full-dataset prediction (with optional tau-adjusted priors)
# --------------------------------------------------------------------------- #

def predict_all_samples(
    latent_df,
    latent_cols: list[str],
    clf_or_bundle,
    pred_col_name: str = "pred_label",
    *,
    use_tau_adjustment: bool = False,
    tau: Optional[float] = None,
) -> pd.DataFrame:
    """
    Predict labels for every row in *latent_df*.

    Parameters
    ----------
    latent_df : DataFrame or path to CSV
        Must contain the columns in *latent_cols*.
    latent_cols : list of str
        Feature column names.
    clf_or_bundle
        A trained classifier **or** a dict bundle with keys
        ``"model"``, ``"classes"``, ``"priors"``, ``"tau"``.
    pred_col_name : str
        Name of the new prediction column added to the output DataFrame.
    use_tau_adjustment : bool
        Apply log-prior correction to predicted probabilities.
    tau : float, optional
        Overrides the tau stored in a bundle.

    Returns
    -------
    DataFrame — a copy of *latent_df* with ``pred_col_name`` and per-class
    probability columns appended.
    """
    latent_df = _load_df(latent_df).copy()

    X = latent_df[latent_cols].apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any():
        raise ValueError("Some feature values became NaN after numeric conversion.")
    X = X.values

    if isinstance(clf_or_bundle, dict):
        clf = clf_or_bundle["model"]
        classes = np.asarray(clf_or_bundle["classes"])
        priors = np.asarray(clf_or_bundle["priors"], dtype=float)
        tau_final = clf_or_bundle.get("tau", 0.0) if tau is None else tau
    else:
        clf = clf_or_bundle
        classes = np.arange(max(clf.classes_) + 1) if hasattr(clf, "classes_") else None
        priors = None
        tau_final = 0.0 if tau is None else tau

    proba = clf.predict_proba(X)

    if use_tau_adjustment and priors is not None and tau_final > 0:
        adjusted = np.log(proba + 1e-12) - tau_final * np.log(priors + 1e-12)
        preds = classes[np.argmax(adjusted, axis=1)]
    else:
        preds = clf.predict(X)

    latent_df[pred_col_name] = preds

    for i in range(proba.shape[1]):
        latent_df[f"{pred_col_name}_prob_{i}"] = proba[:, i]

    return latent_df
