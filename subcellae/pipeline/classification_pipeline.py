"""
classification_pipeline.py
==========================
LightGBM classification on latent features from the AE pipeline.

Reads ``latents.csv`` produced by :func:`run_ae_pipeline`, merges with an
optional external label CSV, and runs a supervised LightGBM classifier.

Outputs saved to ``out_dir``:
  - ``metrics.txt``                  – classification report + summary metrics
  - ``metrics.csv``                  – per-class precision / recall / F1
  - ``confusion_matrix_counts.png``  – raw counts heatmap
  - ``confusion_matrix_norm.png``    – row-normalised heatmap
  - ``feature_importance.png``       – LightGBM split + gain importance
  - ``prob_by_true_class.png``       – predicted max-probability by true class
  - ``classification_results.csv``   – all rows with predicted label + proba
  - ``lgbm_model.pkl``               – fitted LightGBM classifier
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

log = logging.getLogger(__name__)

# Matches the coordinate block that starts with _f<digits>x… or -f<digits>x…
# Used to normalise latents filenames to label-CSV (unique_ID) style.
# latents : control_f0001x0112y0496ps32.tif   (underscore before f)
# label   : control-f0001x0112y0496ps32.tif   (hyphen   before f)
_UNDERSCORE_F = re.compile(r'_(f\d+x\d+y\d+ps\d+\.tiff?)$', re.IGNORECASE)


def _to_unique_id(filename: str) -> str:
    """Convert a latents filename to the label-CSV unique_ID style.

    Replaces the underscore immediately before the coordinate block
    ``f{N}x{X}y{Y}ps{P}.tif`` with a hyphen so it matches unique_ID values
    in the label CSV.

    Example::

        control_f0001x0112y0496ps32.tif  →  control-f0001x0112y0496ps32.tif
    """
    return _UNDERSCORE_F.sub(r'-\1', Path(filename).name)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationConfig:
    """All parameters for one classification run.

    Parameters
    ----------
    latents_csv : Path
        ``latents.csv`` produced by :func:`run_ae_pipeline`.
    out_dir : Path
        Directory where all outputs are saved.

    Label source
    ------------
    label_col : str
        Column to use as the class label.  If the column is already present
        in ``latents.csv`` (e.g. ``"annotation_label_name"``), it is used
        directly.  If not, an external ``label_csv`` is merged in.
    label_csv : str
        Path to an external CSV / Excel with per-patch labels.
        Leave empty to use labels already embedded in ``latents.csv``.
    filename_col : str
        Column to merge on (must exist in both CSVs if ``label_csv`` is set).
    label_order : list[str] or None
        Ordered class names for encoding and plotting.  ``None`` → inferred.
    exclude_labels : list[str] or None
        Labels to drop before training (e.g. ``["No adhesion"]``).

    Features
    --------
    feature_cols : list[str] or None
        Explicit feature column names.  ``None`` → auto-detect all ``z_*``
        columns from ``latents.csv``.
    include_mean_intensity : bool
        Append ``mean_intensity`` to the feature set.

    Split strategy
    --------------
    split_strategy : str
        ``"from_csv"``   – use the ``split`` column already in ``latents.csv``
                           (group-aware, consistent with AE training).
        ``"stratified"`` – fresh stratified random split.
    test_size : float
        Fraction held out for validation (only used for ``"stratified"``).
    random_state : int

    LightGBM
    --------
    n_estimators : int
    learning_rate : float
    num_leaves : int
    min_child_samples : int
    class_weight : str
        ``"balanced"`` or ``None``.
    n_cv_folds : int
        If > 1, run stratified k-fold CV and report mean ± std metrics in
        addition to the final held-out evaluation.
    """

    # --- required ---
    latents_csv: Path
    out_dir: Path

    # --- label source ---
    label_col: str        = "annotation_label_name"
    label_csv: str        = ""
    filename_col: str     = "filename"
    label_order: list     = None
    exclude_labels: list  = None

    # --- features ---
    feature_cols: list           = None
    include_mean_intensity: bool = False

    # --- split ---
    split_strategy: str  = "from_csv"   # "from_csv" | "stratified"
    test_size: float     = 0.2
    random_state: int    = 42

    # --- LightGBM ---
    n_estimators: int       = 500
    learning_rate: float    = 0.05
    num_leaves: int         = 31
    min_child_samples: int  = 20
    class_weight: str       = "balanced"   # "balanced" or null/""
    n_cv_folds: int         = 0            # 0 = no CV

    def __post_init__(self):
        self.latents_csv = Path(self.latents_csv)
        self.out_dir     = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.split_strategy not in {"from_csv", "stratified"}:
            raise ValueError(
                f"split_strategy must be 'from_csv' or 'stratified', "
                f"got {self.split_strategy!r}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_lgbm(cfg: ClassificationConfig, n_classes: int):
    """Instantiate a LightGBM classifier from config."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm is required: pip install lightgbm")

    objective = "multiclass" if n_classes > 2 else "binary"
    cw = cfg.class_weight if cfg.class_weight else None
    return lgb.LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        class_weight=cw,
        objective=objective,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def _evaluate(clf, X, y, class_names) -> dict:
    y_pred  = clf.predict(X)
    y_proba = clf.predict_proba(X)
    return {
        "y_pred":             y_pred,
        "y_proba":            y_proba,
        "accuracy":           accuracy_score(y, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y, y_pred),
        "f1_macro":           f1_score(y, y_pred, average="macro",    zero_division=0),
        "f1_weighted":        f1_score(y, y_pred, average="weighted", zero_division=0),
        "report":             classification_report(y, y_pred,
                                                    target_names=class_names,
                                                    zero_division=0),
        "confusion_matrix":   confusion_matrix(y, y_pred),
    }


def _plot_confusion_matrix(cm, class_names, title, save_path, normalize=False):
    if normalize:
        with np.errstate(all="ignore"):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_feature_importance(clf, feature_names, save_path):
    try:
        import lightgbm as lgb
    except ImportError:
        return

    imp_split = clf.booster_.feature_importance(importance_type="split")
    imp_gain  = clf.booster_.feature_importance(importance_type="gain")

    df = pd.DataFrame({
        "feature": feature_names,
        "split":   imp_split,
        "gain":    imp_gain,
    }).sort_values("gain", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(feature_names) * 0.45)))
    for ax, col, color in zip(axes, ["split", "gain"], ["steelblue", "darkorange"]):
        ax.barh(df["feature"], df[col], color=color)
        ax.set_title(f"Feature importance ({col})")
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_prob_by_class(y_true, y_proba, class_names, save_path, kind="box"):
    max_prob = y_proba.max(axis=1)
    rows = [{"true_class": class_names[yt], "max_prob": mp}
            for yt, mp in zip(y_true, max_prob)]
    df = pd.DataFrame(rows)
    order = [c for c in class_names if c in df["true_class"].values]

    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.5), 4))
    if kind == "violin":
        sns.violinplot(data=df, x="true_class", y="max_prob",
                       order=order, ax=ax, inner="box")
    else:
        sns.boxplot(data=df, x="true_class", y="max_prob", order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_title("Predicted max probability by true class")
    ax.set_xlabel("True class")
    ax.set_ylabel("Max predicted probability")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_per_class_f1(report_dict, class_names, save_path):
    f1s = [report_dict.get(c, {}).get("f1-score", 0.0) for c in class_names]
    fig, ax = plt.subplots(figsize=(max(5, len(class_names) * 1.2), 4))
    bars = ax.bar(class_names, f1s, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 score (validation set)")
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def run_classification_pipeline(cfg: ClassificationConfig) -> dict:
    """Run the full LightGBM classification pipeline.

    Parameters
    ----------
    cfg : ClassificationConfig

    Returns
    -------
    dict with keys ``clf``, ``metrics``, ``df_results``.
    """
    log.info("=" * 60)
    log.info("Classification Pipeline  (LightGBM)")
    log.info("  latents_csv    : %s", cfg.latents_csv)
    log.info("  out_dir        : %s", cfg.out_dir)
    log.info("  label_col      : %s", cfg.label_col)
    log.info("  split_strategy : %s", cfg.split_strategy)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load and merge data
    # ------------------------------------------------------------------
    log.info("Step 1: Loading data …")
    df = pd.read_csv(cfg.latents_csv)

    # Merge external label CSV if provided and label_col not already in df
    if cfg.label_csv and (cfg.label_col not in df.columns or
                          df[cfg.label_col].isna().all()):
        label_path = Path(cfg.label_csv)
        ext = label_path.suffix.lower()
        ldf = (pd.read_excel(label_path)
               if ext in {".xlsx", ".xls"}
               else pd.read_csv(label_path))

        # Normalise label-CSV join key to bare basename
        ldf["_uid"] = ldf[cfg.filename_col].astype(str).apply(
            lambda p: Path(p).name
        )

        # Convert latents filenames (underscore style) → label-CSV style (hyphen)
        # e.g. control_f0001x0112y0496ps32.tif → control-f0001x0112y0496ps32.tif
        df["_uid"] = df["filename"].apply(_to_unique_id)

        n_before = len(df)
        df = df.merge(
            ldf[["_uid", cfg.label_col]].drop_duplicates("_uid"),
            on="_uid", how="left",
            suffixes=("", "_ext"),
        )
        matched = df[cfg.label_col].notna().sum()
        log.info("  Merged external label CSV: %s", label_path)
        log.info("  Matched %d / %d rows  (unmatched will be dropped as unlabelled)",
                 matched, n_before)
        if matched == 0:
            log.warning(
                "  No rows matched! Check that filename formats align.\n"
                "  latents example : %s → normalised: %s\n"
                "  label   example : %s",
                df["filename"].iloc[0] if len(df) else "?",
                df["_uid"].iloc[0] if len(df) else "?",
                ldf["_uid"].iloc[0] if len(ldf) else "?",
            )
        df = df.drop(columns=["_uid"])

    # Filter to labelled rows.
    # When using an external label_csv the annotation_label column (which
    # marks patches NOT in the training annotation as -1) must NOT be used
    # to filter — those patches can still have valid external labels.
    if not cfg.label_csv and "annotation_label" in df.columns:
        df = df[df["annotation_label"] != -1].copy()
    df = df[df[cfg.label_col].notna()].copy()
    log.info("  Labelled rows after filter: %d", len(df))

    # Exclude specified labels
    if cfg.exclude_labels:
        df = df[~df[cfg.label_col].isin(cfg.exclude_labels)].reset_index(drop=True)
        log.info("  After excluding %s: %d rows", cfg.exclude_labels, len(df))

    # Resolve label order and encode
    label_order = cfg.label_order or sorted(df[cfg.label_col].dropna().unique().tolist())
    label_to_id = {lbl: i for i, lbl in enumerate(label_order)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    df["_y"] = df[cfg.label_col].map(label_to_id)
    df = df.dropna(subset=["_y"]).copy()
    df["_y"] = df["_y"].astype(int)
    n_classes = len(label_order)
    log.info("  Classes (%d): %s", n_classes, label_order)

    # ------------------------------------------------------------------
    # 2. Feature matrix
    # ------------------------------------------------------------------
    log.info("Step 2: Building feature matrix …")
    if cfg.feature_cols:
        feat_cols = cfg.feature_cols
    else:
        feat_cols = [c for c in df.columns if c.startswith("z_")]

    if cfg.include_mean_intensity and "mean_intensity" in df.columns:
        feat_cols = feat_cols + ["mean_intensity"]

    log.info("  Features: %d  (%s … %s)", len(feat_cols), feat_cols[0], feat_cols[-1])
    X = df[feat_cols].values.astype(np.float32)
    y = df["_y"].values

    # ------------------------------------------------------------------
    # 3. Train / val split
    # ------------------------------------------------------------------
    log.info("Step 3: Splitting data (strategy=%r) …", cfg.split_strategy)
    if cfg.split_strategy == "from_csv" and "split" in df.columns:
        train_mask = df["split"].values == "train"
        val_mask   = df["split"].values == "val"
        X_train, y_train = X[train_mask], y[train_mask]
        X_val,   y_val   = X[val_mask],   y[val_mask]
        df_train = df[train_mask].copy()
        df_val   = df[val_mask].copy()
    else:
        idx = np.arange(len(X))
        tr_idx, va_idx = train_test_split(
            idx, test_size=cfg.test_size, stratify=y,
            random_state=cfg.random_state,
        )
        X_train, y_train = X[tr_idx], y[tr_idx]
        X_val,   y_val   = X[va_idx], y[va_idx]
        df_train = df.iloc[tr_idx].copy()
        df_val   = df.iloc[va_idx].copy()

    log.info("  Train: %d  |  Val: %d", len(X_train), len(X_val))

    # ------------------------------------------------------------------
    # 4. Optional cross-validation
    # ------------------------------------------------------------------
    cv_results = {}
    if cfg.n_cv_folds > 1:
        log.info("Step 4: %d-fold stratified CV …", cfg.n_cv_folds)
        skf = StratifiedKFold(n_splits=cfg.n_cv_folds, shuffle=True,
                              random_state=cfg.random_state)
        fold_acc, fold_bacc, fold_f1 = [], [], []
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            clf_cv = _build_lgbm(cfg, n_classes)
            clf_cv.fit(X[tr], y[tr])
            ev = _evaluate(clf_cv, X[va], y[va], label_order)
            fold_acc.append(ev["accuracy"])
            fold_bacc.append(ev["balanced_accuracy"])
            fold_f1.append(ev["f1_macro"])
            log.info("  Fold %d – acc=%.3f  bacc=%.3f  f1_macro=%.3f",
                     fold, ev["accuracy"], ev["balanced_accuracy"], ev["f1_macro"])
        cv_results = {
            "cv_accuracy_mean":  float(np.mean(fold_acc)),
            "cv_accuracy_std":   float(np.std(fold_acc)),
            "cv_bacc_mean":      float(np.mean(fold_bacc)),
            "cv_bacc_std":       float(np.std(fold_bacc)),
            "cv_f1_macro_mean":  float(np.mean(fold_f1)),
            "cv_f1_macro_std":   float(np.std(fold_f1)),
        }
        log.info("  CV summary – acc=%.3f±%.3f  bacc=%.3f±%.3f  f1=%.3f±%.3f",
                 cv_results["cv_accuracy_mean"],  cv_results["cv_accuracy_std"],
                 cv_results["cv_bacc_mean"],      cv_results["cv_bacc_std"],
                 cv_results["cv_f1_macro_mean"],  cv_results["cv_f1_macro_std"])
    else:
        log.info("Step 4: CV skipped (n_cv_folds=%d)", cfg.n_cv_folds)

    # ------------------------------------------------------------------
    # 5. Train final model on train split
    # ------------------------------------------------------------------
    log.info("Step 5: Training final LightGBM model …")
    clf = _build_lgbm(cfg, n_classes)
    clf.fit(X_train, y_train)
    joblib.dump(clf, str(cfg.out_dir / "lgbm_model.pkl"))
    log.info("  Model saved → %s", cfg.out_dir / "lgbm_model.pkl")

    # ------------------------------------------------------------------
    # 6. Evaluate on validation set
    # ------------------------------------------------------------------
    log.info("Step 6: Evaluating on validation set …")
    metrics = _evaluate(clf, X_val, y_val, label_order)
    log.info("  Accuracy          : %.4f", metrics["accuracy"])
    log.info("  Balanced accuracy : %.4f", metrics["balanced_accuracy"])
    log.info("  F1 macro          : %.4f", metrics["f1_macro"])
    log.info("  F1 weighted       : %.4f", metrics["f1_weighted"])
    log.info("\n%s", metrics["report"])

    # Save metrics text
    summary_lines = [
        f"accuracy          : {metrics['accuracy']:.4f}",
        f"balanced_accuracy : {metrics['balanced_accuracy']:.4f}",
        f"f1_macro          : {metrics['f1_macro']:.4f}",
        f"f1_weighted       : {metrics['f1_weighted']:.4f}",
    ]
    if cv_results:
        summary_lines += [
            "",
            f"cv_accuracy       : {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}",
            f"cv_bacc           : {cv_results['cv_bacc_mean']:.4f} ± {cv_results['cv_bacc_std']:.4f}",
            f"cv_f1_macro       : {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}",
        ]
    summary_lines += ["", "Classification report (validation):", metrics["report"]]
    (cfg.out_dir / "metrics.txt").write_text("\n".join(summary_lines))

    # Save per-class metrics CSV
    from sklearn.metrics import classification_report as _cr
    report_dict = _cr(y_val, metrics["y_pred"],
                      target_names=label_order,
                      zero_division=0, output_dict=True)
    metrics_rows = [
        {
            "class": c,
            "precision": report_dict[c]["precision"],
            "recall":    report_dict[c]["recall"],
            "f1":        report_dict[c]["f1-score"],
            "support":   report_dict[c]["support"],
        }
        for c in label_order if c in report_dict
    ]
    pd.DataFrame(metrics_rows).to_csv(str(cfg.out_dir / "metrics.csv"), index=False)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    log.info("Step 7: Saving plots …")

    # Confusion matrices
    acc = metrics["accuracy"]
    _plot_confusion_matrix(
        metrics["confusion_matrix"], label_order,
        title=f"Confusion matrix (counts)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_counts.png",
        normalize=False,
    )
    _plot_confusion_matrix(
        metrics["confusion_matrix"], label_order,
        title=f"Confusion matrix (normalised)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_norm.png",
        normalize=True,
    )

    # Feature importance
    _plot_feature_importance(clf, feat_cols,
                             cfg.out_dir / "feature_importance.png")

    # Per-class F1 bar chart
    _plot_per_class_f1(report_dict, label_order,
                       cfg.out_dir / "f1_per_class.png")

    # Max predicted probability by true class
    _plot_prob_by_class(y_val, metrics["y_proba"], label_order,
                        cfg.out_dir / "prob_by_true_class.png")

    # ------------------------------------------------------------------
    # 8. Save full results CSV (predictions for all labelled rows)
    # ------------------------------------------------------------------
    log.info("Step 8: Saving classification_results.csv …")
    all_pred  = clf.predict(X)
    all_proba = clf.predict_proba(X)

    df_out = df[["filename", "filepath", "condition_name", "group", "split",
                 cfg.label_col]].copy()
    df_out["pred_label"] = [id_to_label[p] for p in all_pred]
    df_out["correct"]    = df_out[cfg.label_col] == df_out["pred_label"]
    for i, lbl in enumerate(label_order):
        df_out[f"prob_{lbl}"] = all_proba[:, i]

    df_out.to_csv(str(cfg.out_dir / "classification_results.csv"), index=False)

    log.info("Classification pipeline complete.  Outputs → %s", cfg.out_dir)
    return {
        "clf":        clf,
        "metrics":    metrics,
        "cv_results": cv_results,
        "df_results": df_out,
        "label_order": label_order,
    }
