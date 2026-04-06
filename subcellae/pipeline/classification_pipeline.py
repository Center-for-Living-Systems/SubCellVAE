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
  - ``classification_results.csv``   – labelled rows with predicted label + proba
  - ``model.pkl``                    – fitted classifier (lgbm / svm / mlp / knn)
  - ``umap_predicted_label.png``     – UMAP of ALL patches, coloured by predicted label (tab10)
  - ``umap_true_label.png``          – UMAP of ALL patches, coloured by true label (where available)
  - ``umap_split.png``               – train (●) + val (▲) overlaid, coloured by predicted label
  - ``umap_split_fa4.png``           – same, top-4 classes only (excludes "No adhesion")
  - ``umap_split_train.png``         – train patches only, all classes
  - ``umap_split_train_fa4.png``     – train patches only, top-4 classes
  - ``umap_split_val.png``           – val patches only, all classes
  - ``umap_split_val_fa4.png``       – val patches only, top-4 classes
  - ``umap_all_model.pkl``           – fitted UMAP model (all patches)
  - ``patch_sort/gt{x}pred{y}/``     – patches copied by true-class / predicted-class index;
                                       unlabelled patches go into ``gtnpred{y}/``
"""

from __future__ import annotations

import logging
import re
import shutil
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
    exclude_labels: list  = None         # dropped before training AND evaluation
    metrics_exclude_labels: list = None  # kept in training, excluded from reported metrics only

    # --- features ---
    feature_cols: list           = None
    include_mean_intensity: bool = False

    # --- split ---
    split_strategy: str  = "from_csv"   # "from_csv" | "stratified"
    test_size: float     = 0.2
    random_state: int    = 42

    # --- classifier selection ---
    # "lgbm" | "svm" | "mlp" | "knn"
    classifier_type: str = "lgbm"

    # --- LightGBM ---
    n_estimators: int       = 500
    learning_rate: float    = 0.05
    num_leaves: int         = 31
    min_child_samples: int  = 20
    class_weight: str       = "balanced"   # "balanced" or null/""
    n_cv_folds: int         = 0            # 0 = no CV

    # --- SVM (used when classifier_type = "svm") ---
    svm_C: float     = 10.0    # regularisation; larger = tighter fit
    svm_gamma: str   = "scale" # "scale" | "auto" | float

    # --- MLP (used when classifier_type = "mlp") ---
    mlp_hidden_layers: list = None   # e.g. [64, 32]; None → [64, 32] default
    mlp_max_iter: int       = 1000

    # --- KNN (used when classifier_type = "knn") ---
    knn_n_neighbors: int = 10

    # --- distance features ---
    # List of patch-prep plot/csv directories (one per condition).
    # The pipeline will find the latest data_prep_record_*.csv in each dir,
    # concatenate them, and merge d00..d07 + equiv_diam into the latents df.
    # Leave as None / [] to use only latent features.
    dist_patch_prep_dirs: list = None
    # Multiplicative weight applied to all d* columns before classification
    # and UMAP.  Latent features are always kept at weight 1.0.
    # Use ~100 to bring normalised distances (ratio scale) into a similar
    # numerical range as the latent dimensions.
    dist_feature_weight: float = 100.0

    # --- patch sorting ---
    sort_labelled: bool   = True    # copy train/val patches into gt{x}pred{y} folders
    sort_unlabelled: bool = False   # copy unlabelled patches into test/gtnpred{y} folders

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


def _build_classifier(cfg: ClassificationConfig, n_classes: int):
    """Instantiate a classifier from config based on classifier_type."""
    ctype = cfg.classifier_type.lower()

    if ctype == "lgbm":
        return _build_lgbm(cfg, n_classes)

    if ctype == "svm":
        from sklearn.svm import SVC
        cw = cfg.class_weight if cfg.class_weight else None
        return SVC(
            kernel="rbf",
            C=cfg.svm_C,
            gamma=cfg.svm_gamma,
            class_weight=cw,
            probability=True,   # needed for predict_proba
            random_state=cfg.random_state,
        )

    if ctype == "mlp":
        from sklearn.neural_network import MLPClassifier
        hidden = tuple(cfg.mlp_hidden_layers) if cfg.mlp_hidden_layers else (64, 32)
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=cfg.mlp_max_iter,
            random_state=cfg.random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )

    if ctype == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=cfg.knn_n_neighbors,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier_type: {cfg.classifier_type!r}. "
                     "Choose from: lgbm, svm, mlp, knn")


def _evaluate(clf, X, y, class_names) -> dict:
    y_pred  = clf.predict(X)
    y_proba = clf.predict_proba(X)
    # Restrict all metrics to the expected class indices (len of class_names).
    # When the classifier was trained on more classes than class_names covers
    # (e.g. metrics_exclude_labels filtered some classes from y but not y_pred),
    # labels= ensures spurious predictions don't break the report.
    labels = list(range(len(class_names)))
    return {
        "y_pred":             y_pred,
        "y_proba":            y_proba,
        "accuracy":           accuracy_score(y, y_pred),
        "balanced_accuracy":  balanced_accuracy_score(y, y_pred, adjusted=False),
        "f1_macro":           f1_score(y, y_pred, average="macro",    labels=labels, zero_division=0),
        "f1_weighted":        f1_score(y, y_pred, average="weighted", labels=labels, zero_division=0),
        "report":             classification_report(y, y_pred,
                                                    labels=labels,
                                                    target_names=class_names,
                                                    zero_division=0),
        "confusion_matrix":   confusion_matrix(y, y_pred, labels=labels),
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
# UMAP and patch-sorting helpers
# ---------------------------------------------------------------------------

def _plot_umap_predicted(
    emb: np.ndarray,
    pred_names: list,
    label_order: list,
    title: str,
    save_path: Path,
    label_to_color: dict | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
):
    """Scatter plot of a 2-D embedding coloured by label.

    Parameters
    ----------
    label_to_color : dict, optional
        Mapping of label name → hex colour string.  Falls back to tab10
        palette by index if not provided or if a label is missing.
    xlim, ylim : (float, float), optional
        Fixed axis limits.  Pass the bounds computed from the full embedding
        so that all subset plots share the same coordinate system.
    """
    palette = plt.get_cmap("tab10")
    pred_arr = np.array(pred_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lbl in enumerate(label_order):
        mask = pred_arr == lbl
        if not mask.any():
            continue
        if label_to_color and lbl in label_to_color:
            color = label_to_color[lbl]
        else:
            color = palette(i / max(len(label_order) - 1, 1))
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            label=lbl, s=4, alpha=0.6, color=color,
        )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_umap_split(
    emb: np.ndarray,
    split_labels: np.ndarray,
    pred_names: np.ndarray,
    label_order: list,
    title: str,
    save_path: Path,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
):
    """UMAP coloured by train/val split, with FA-type shown by marker shape.

    train patches → circles (o)
    val   patches → triangles (^)

    This diagnostic reveals whether the class cluster structure seen for
    training patches also holds for held-out validation patches.
    """
    palette = plt.get_cmap("tab10")
    split_arr = np.array(split_labels)
    pred_arr  = np.array(pred_names)

    split_styles = {
        "train": ("o", 0.5, 4,  "train"),
        "val":   ("^", 0.9, 12, "val"),
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lbl in enumerate(label_order):
        color = palette(i / max(len(label_order) - 1, 1))
        for sp, (marker, alpha, size, sp_label) in split_styles.items():
            mask = (pred_arr == lbl) & (split_arr == sp)
            if not mask.any():
                continue
            legend_label = f"{lbl} [{sp_label}]" if sp == "val" else lbl
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                label=legend_label,
                s=size, alpha=alpha, color=color,
                marker=marker,
                linewidths=0,
            )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _sort_patches_to_folders(
    df_all: pd.DataFrame,
    pred_names: list,
    label_order: list,
    label_col: str,
    sort_dir: Path,
    sort_labelled: bool = True,
    sort_unlabelled: bool = False,
):
    """Copy patch tifs into a three-level folder hierarchy.

    Structure
    ---------
    ::

        patch_sort/
          train/             labelled patches whose AE split == "train"
            gt{x}pred{y}/
          val/               labelled patches whose AE split == "val"
            gt{x}pred{y}/
          test/              unlabelled patches (no ground-truth label)
            gtnpred{y}/

    The relevant split-level folders are deleted and recreated before each run
    so the results are always fresh.  ``sort_labelled`` controls train/val;
    ``sort_unlabelled`` controls test.
    Files are copied (originals are preserved).
    """
    if not sort_labelled and not sort_unlabelled:
        log.info("  Patch sorting disabled (both sort_labelled and sort_unlabelled are false)")
        return

    sort_dir.mkdir(parents=True, exist_ok=True)

    # Delete stale split folders before writing fresh results
    for split_folder, enabled in [("train", sort_labelled),
                                   ("val",   sort_labelled),
                                   ("test",  sort_unlabelled)]:
        target = sort_dir / split_folder
        if enabled and target.exists():
            shutil.rmtree(str(target))
            log.debug("  Cleared stale folder: %s", target)

    label_to_id = {lbl: i for i, lbl in enumerate(label_order)}

    missing_files = 0
    for i, (_, row) in enumerate(df_all.iterrows()):
        pred_idx = label_to_id.get(pred_names[i], "?")

        gt_val = row.get(label_col, None)
        has_label = (
            gt_val is not None
            and not (isinstance(gt_val, float) and np.isnan(gt_val))
            and str(gt_val) != ""
        )

        if has_label:
            if not sort_labelled:
                continue
            gt_str = str(gt_val)
            if gt_str not in label_to_id:
                # Label exists but is not in label_order (e.g. excluded class) — skip
                continue
            gt_key = label_to_id[gt_str]
            split_val = str(row.get("split", "train"))
            split_folder = split_val if split_val in {"train", "val"} else "train"
            folder_name = f"gt{gt_key}pred{pred_idx}"
        else:
            if not sort_unlabelled:
                continue
            split_folder = "test"
            folder_name = f"gtnpred{pred_idx}"

        dest_dir = sort_dir / split_folder / folder_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        src = Path(str(row["filepath"]))
        if not src.exists():
            missing_files += 1
            continue
        shutil.copy2(str(src), str(dest_dir / src.name))

    if missing_files:
        log.warning("  %d source files not found and were skipped", missing_files)


# ---------------------------------------------------------------------------
# Distance-feature loader
# ---------------------------------------------------------------------------

def _load_dist_csvs(dirs: list) -> pd.DataFrame:
    """Load and concatenate patch-prep record CSVs from one or more directories.

    For each directory, finds all files matching
    ``data_prep_record_*_to_<N>.csv`` and picks the one with the **largest**
    ``<N>`` (most complete run).  The selected files are concatenated and the
    result is returned.

    The returned DataFrame has at least ``crop_img_filename``, ``equiv_diam``,
    and ``d00``…``d{N-1}`` columns.
    """
    import glob as _glob

    dfs = []
    for d in dirs:
        d = Path(d)
        candidates = list(d.glob("data_prep_record_*_to_*.csv"))
        if not candidates:
            log.warning("  No data_prep_record_*_to_*.csv found in %s – skipping", d)
            continue

        # Extract the trailing integer (the 'to_NNN' part) for each file
        def _end_idx(p: Path) -> int:
            stem = p.stem                        # e.g. data_prep_record_control_ch1_f_0_to_42
            last = stem.rsplit("_to_", 1)[-1]
            try:
                return int(last)
            except ValueError:
                return -1

        best = max(candidates, key=_end_idx)
        log.info("  Dist CSV: %s", best)
        dfs.append(pd.read_csv(best))

    if not dfs:
        raise FileNotFoundError(
            "No data_prep_record CSVs found in any of the provided "
            f"dist_patch_prep_dirs: {dirs}"
        )
    return pd.concat(dfs, ignore_index=True)


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
    # 1b. Merge distance features (optional)
    # ------------------------------------------------------------------
    dist_cols: list[str] = []
    dist_df_slim = None          # kept in scope for reuse in step 9
    if cfg.dist_patch_prep_dirs:
        log.info("Step 1b: Loading distance features from %d dir(s) …",
                 len(cfg.dist_patch_prep_dirs))
        dist_df = _load_dist_csvs(cfg.dist_patch_prep_dirs)

        # dist CSV join key is crop_img_filename (underscore style, same as latents filename)
        dist_df["_dist_key"] = dist_df["crop_img_filename"].apply(
            lambda p: Path(p).name
        )
        df["_dist_key"] = df["filename"].apply(lambda p: Path(p).name)

        dist_cols = ["equiv_diam"] + [c for c in dist_df.columns
                                       if re.fullmatch(r"d\d{2}", c)]
        dist_df_slim = dist_df[["_dist_key"] + dist_cols].drop_duplicates("_dist_key")

        n_before = len(df)
        df = df.merge(dist_df_slim, on="_dist_key", how="left")
        df = df.drop(columns=["_dist_key"])
        matched = df[dist_cols[0]].notna().sum()
        log.info("  Merged dist features: %d / %d rows matched", matched, n_before)
        if matched == 0:
            log.warning("  No distance features matched — check that patch_prep_dirs "
                        "contain records for the same patches as latents_csv.")

    # ------------------------------------------------------------------
    # 2. Feature matrix
    # ------------------------------------------------------------------
    log.info("Step 2: Building feature matrix …")
    if cfg.feature_cols:
        feat_cols = list(cfg.feature_cols)
    else:
        feat_cols = [c for c in df.columns if c.startswith("z_")]

    if dist_cols:
        d_feats = [c for c in dist_cols if c.startswith("d")]   # d00..d07; exclude equiv_diam
        feat_cols = feat_cols + [c for c in d_feats if c in df.columns]
        log.info("  Distance features added: %s", d_feats)

    if cfg.include_mean_intensity and "mean_intensity" in df.columns:
        feat_cols = feat_cols + ["mean_intensity"]

    # Build per-feature scale: dist cols × dist_feature_weight, rest × 1.0
    _dist_feat_set = set(c for c in feat_cols if re.fullmatch(r"d\d{2}", c))
    feat_scale = np.array(
        [cfg.dist_feature_weight if c in _dist_feat_set else 1.0 for c in feat_cols],
        dtype=np.float32,
    )
    if _dist_feat_set:
        log.info("  Feature scaling: latent ×1  dist ×%.0f  (dist cols: %s)",
                 cfg.dist_feature_weight, sorted(_dist_feat_set))

    log.info("  Features: %d  (%s … %s)", len(feat_cols), feat_cols[0], feat_cols[-1])
    X = df[feat_cols].values.astype(np.float32) * feat_scale
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
            clf_cv = _build_classifier(cfg, n_classes)
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
    log.info("Step 5: Training final %s model …", cfg.classifier_type.upper())
    clf = _build_classifier(cfg, n_classes)
    clf.fit(X_train, y_train)
    joblib.dump(clf, str(cfg.out_dir / "model.pkl"))
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

    # Optional: metrics excluding certain labels from the reported numbers.
    # Those labels were still in training; only the evaluation rows are filtered.
    metrics_excl = None
    excl_labels    = cfg.metrics_exclude_labels or []
    excl_ids       = {label_to_id[l] for l in excl_labels if l in label_to_id}
    filtered_order = [l for l in label_order if l not in excl_labels]
    if excl_ids and filtered_order:
        keep = ~np.isin(y_val, list(excl_ids))
        if keep.any():
            # Remap class ids so they are 0-based for the filtered label list
            kept_ids   = sorted(label_to_id[l] for l in filtered_order)
            id_remap   = {old: new for new, old in enumerate(kept_ids)}
            y_val_filt = np.array([id_remap[v] for v in y_val[keep]])
            metrics_excl = _evaluate(clf, X_val[keep], y_val_filt, filtered_order)
            log.info("  [excl %s] Accuracy : %.4f  Balanced acc : %.4f  F1 macro : %.4f",
                     excl_labels,
                     metrics_excl["accuracy"],
                     metrics_excl["balanced_accuracy"],
                     metrics_excl["f1_macro"])

    # Save metrics text
    summary_lines = [
        f"accuracy          : {metrics['accuracy']:.4f}",
        f"balanced_accuracy : {metrics['balanced_accuracy']:.4f}",
        f"f1_macro          : {metrics['f1_macro']:.4f}",
        f"f1_weighted       : {metrics['f1_weighted']:.4f}",
    ]
    if metrics_excl is not None:
        summary_lines += [
            "",
            f"# --- metrics excluding {excl_labels} (kept in training) ---",
            f"accuracy_excl     : {metrics_excl['accuracy']:.4f}",
            f"balanced_acc_excl : {metrics_excl['balanced_accuracy']:.4f}",
            f"f1_macro_excl     : {metrics_excl['f1_macro']:.4f}",
        ]
    if cv_results:
        summary_lines += [
            "",
            f"cv_accuracy       : {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}",
            f"cv_bacc           : {cv_results['cv_bacc_mean']:.4f} ± {cv_results['cv_bacc_std']:.4f}",
            f"cv_f1_macro       : {cv_results['cv_f1_macro_mean']:.4f} ± {cv_results['cv_f1_macro_std']:.4f}",
        ]
    summary_lines += ["", "Classification report (validation):", metrics["report"]]
    if metrics_excl is not None:
        summary_lines += [
            "",
            f"Classification report (validation, excl. {excl_labels}):",
            metrics_excl["report"],
        ]
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

    # Confusion matrices — validation set
    acc = metrics["accuracy"]
    _plot_confusion_matrix(
        metrics["confusion_matrix"], label_order,
        title=f"Val – confusion matrix (counts)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_counts_val.png",
        normalize=False,
    )
    _plot_confusion_matrix(
        metrics["confusion_matrix"], label_order,
        title=f"Val – confusion matrix (normalised)  acc={acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_norm_val.png",
        normalize=True,
    )

    # Confusion matrices — training set
    train_metrics = _evaluate(clf, X_train, y_train, label_order)
    train_acc = train_metrics["accuracy"]
    _plot_confusion_matrix(
        train_metrics["confusion_matrix"], label_order,
        title=f"Train – confusion matrix (counts)  acc={train_acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_counts_train.png",
        normalize=False,
    )
    _plot_confusion_matrix(
        train_metrics["confusion_matrix"], label_order,
        title=f"Train – confusion matrix (normalised)  acc={train_acc:.3f}",
        save_path=cfg.out_dir / "confusion_matrix_norm_train.png",
        normalize=True,
    )
    log.info("  Train acc=%.4f  |  Val acc=%.4f", train_acc, acc)

    # Feature importance (LightGBM only)
    if cfg.classifier_type.lower() == "lgbm":
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

    # ------------------------------------------------------------------
    # 9. UMAP of ALL patches coloured by predicted label
    # ------------------------------------------------------------------
    log.info("Step 9: UMAP of all patches coloured by predicted label …")
    try:
        from umap import UMAP as _UMAP

        # Load every row from latents.csv (labelled + unlabelled)
        df_all = pd.read_csv(cfg.latents_csv)

        # Merge distance features into df_all (same join as step 1b)
        if dist_df_slim is not None:
            df_all["_dist_key"] = df_all["filename"].apply(lambda p: Path(p).name)
            df_all = df_all.merge(dist_df_slim, on="_dist_key", how="left").drop(columns=["_dist_key"])

        # Merge external labels if provided (for GT colour in a second plot)
        if cfg.label_csv:
            label_path = Path(cfg.label_csv)
            ext = label_path.suffix.lower()
            ldf_all = (pd.read_excel(label_path)
                       if ext in {".xlsx", ".xls"}
                       else pd.read_csv(label_path))
            ldf_all["_uid"] = ldf_all[cfg.filename_col].astype(str).apply(
                lambda p: Path(p).name
            )
            df_all["_uid"] = df_all["filename"].apply(_to_unique_id)
            df_all = df_all.merge(
                ldf_all[["_uid", cfg.label_col]].drop_duplicates("_uid"),
                on="_uid", how="left",
            ).drop(columns=["_uid"])

        X_all = df_all[feat_cols].values.astype(np.float32) * feat_scale
        log.info("  UMAP input shape: %s  (features: %s)", X_all.shape, feat_cols)
        pred_all       = clf.predict(X_all)
        pred_names_all = [id_to_label[p] for p in pred_all]
        proba_all      = clf.predict_proba(X_all)

        # Save predictions for ALL patches (labelled + unlabelled) so that
        # the cross-classification visualisation pipeline can use them.
        _all_out_cols = ["filename"]
        for _c in ["condition_name", "group", "split"]:
            if _c in df_all.columns:
                _all_out_cols.append(_c)
        if cfg.label_col in df_all.columns:
            _all_out_cols.append(cfg.label_col)
        df_preds_all = df_all[_all_out_cols].copy()
        df_preds_all["pred_label"] = pred_names_all
        df_preds_all["max_prob"]   = proba_all.max(axis=1)
        df_preds_all.to_csv(str(cfg.out_dir / "predictions_all.csv"), index=False)
        log.info("  Saved predictions_all.csv (%d rows)", len(df_preds_all))

        reducer_all = _UMAP(n_components=2, random_state=cfg.random_state)
        emb_all = reducer_all.fit_transform(X_all)
        joblib.dump(reducer_all, str(cfg.out_dir / "umap_all_model.pkl"))

        # Load project colour map (falls back gracefully if import fails)
        try:
            from subcellae.utils.label_colors import (
                classification_label_to_color,
                position_label_to_color,
            )
            _color_maps = {
                "classification": classification_label_to_color,
                "Position":       position_label_to_color,
            }
            color_map = _color_maps.get(cfg.label_col)
        except ImportError:
            color_map = None

        _task_name   = cfg.label_col          # used in plot titles
        pred_arr_all = np.array(pred_names_all)
        fa4_order    = label_order[:4]        # first 4 classes (e.g. mask out uncertain)

        # Compute axis limits from the full embedding with 5% padding
        _pad = 0.05
        x_range = emb_all[:, 0].max() - emb_all[:, 0].min()
        y_range = emb_all[:, 1].max() - emb_all[:, 1].min()
        xlim = (emb_all[:, 0].min() - _pad * x_range,
                emb_all[:, 0].max() + _pad * x_range)
        ylim = (emb_all[:, 1].min() - _pad * y_range,
                emb_all[:, 1].max() + _pad * y_range)

        def _umap_subset(row_mask, class_mask, title, save_path):
            """Helper: plot UMAP for rows in row_mask, classes in class_mask."""
            combined = row_mask & class_mask
            if combined.any():
                _plot_umap_predicted(
                    emb_all[combined],
                    pred_arr_all[combined].tolist(),
                    [l for l in label_order if l in pred_arr_all[combined]],
                    title=title,
                    save_path=save_path,
                    label_to_color=color_map,
                    xlim=xlim,
                    ylim=ylim,
                )

        all_rows    = np.ones(len(df_all), dtype=bool)
        all_classes = np.ones(len(df_all), dtype=bool)
        fa4_mask    = np.isin(pred_arr_all, fa4_order)

        cond_col = "condition_name" if "condition_name" in df_all.columns else None
        ctrl_mask = (df_all[cond_col].values == "control") if cond_col else all_rows
        ycomp_mask = (df_all[cond_col].values == "ycomp")  if cond_col else all_rows

        # ── all patches, all classes ──────────────────────────────────────
        _plot_umap_predicted(
            emb_all, pred_names_all, label_order,
            title=f"UMAP – all patches, predicted {_task_name}",
            save_path=cfg.out_dir / "umap_predicted_all.png",
            label_to_color=color_map,
            xlim=xlim, ylim=ylim,
        )
        # ── all patches, top-4 classes only ──────────────────────────────
        _umap_subset(all_rows, fa4_mask,
                     f"UMAP – all patches, predicted {_task_name} (top-4)",
                     cfg.out_dir / "umap_predicted_all_fa4.png")
        # ── control, all classes ──────────────────────────────────────────
        _umap_subset(ctrl_mask, all_classes,
                     f"UMAP – control, predicted {_task_name}",
                     cfg.out_dir / "umap_predicted_control.png")
        # ── control, top-4 classes only ───────────────────────────────────
        _umap_subset(ctrl_mask, fa4_mask,
                     f"UMAP – control, predicted {_task_name} (top-4)",
                     cfg.out_dir / "umap_predicted_control_fa4.png")
        # ── ycomp, all classes ────────────────────────────────────────────
        _umap_subset(ycomp_mask, all_classes,
                     f"UMAP – ycomp, predicted {_task_name}",
                     cfg.out_dir / "umap_predicted_ycomp.png")
        # ── ycomp, top-4 classes only ─────────────────────────────────────
        _umap_subset(ycomp_mask, fa4_mask,
                     f"UMAP – ycomp, predicted {_task_name} (top-4)",
                     cfg.out_dir / "umap_predicted_ycomp_fa4.png")

        # ── true label — labelled patches only ────────────────────────────
        if cfg.label_col in df_all.columns:
            gt_names_all = np.array(df_all[cfg.label_col].fillna("").tolist())
            labelled_mask = np.isin(gt_names_all, label_order)
            if labelled_mask.any():
                _plot_umap_predicted(
                    emb_all[labelled_mask],
                    gt_names_all[labelled_mask].tolist(),
                    label_order,
                    title=f"UMAP – labelled patches, true {_task_name}",
                    save_path=cfg.out_dir / "umap_true_label.png",
                    label_to_color=color_map,
                    xlim=xlim, ylim=ylim,
                )

        # ── train vs val split coloured by predicted label ────────────────
        if "split" in df_all.columns:
            split_vals = df_all["split"].fillna("unlabelled").values
            train_mask = split_vals == "train"
            val_mask   = split_vals == "val"

            # combined overlay (train=circle, val=triangle)
            _plot_umap_split(
                emb_all,
                split_labels=split_vals,
                pred_names=pred_arr_all,
                label_order=label_order,
                title=f"UMAP – train (●) vs val (▲), predicted {_task_name}",
                save_path=cfg.out_dir / "umap_split.png",
                xlim=xlim, ylim=ylim,
            )
            # combined overlay, top-4 classes only
            _plot_umap_split(
                emb_all[fa4_mask],
                split_labels=split_vals[fa4_mask],
                pred_names=pred_arr_all[fa4_mask],
                label_order=fa4_order,
                title=f"UMAP – train (●) vs val (▲), predicted {_task_name} (top-4)",
                save_path=cfg.out_dir / "umap_split_fa4.png",
                xlim=xlim, ylim=ylim,
            )
            # train patches only, all classes
            _umap_subset(train_mask, all_classes,
                         f"UMAP – train only, predicted {_task_name}",
                         cfg.out_dir / "umap_split_train.png")
            # train patches only, top-4 classes
            _umap_subset(train_mask, fa4_mask,
                         f"UMAP – train only, predicted {_task_name} (top-4)",
                         cfg.out_dir / "umap_split_train_fa4.png")
            # val patches only, all classes
            _umap_subset(val_mask, all_classes,
                         f"UMAP – val only, predicted {_task_name}",
                         cfg.out_dir / "umap_split_val.png")
            # val patches only, top-4 classes
            _umap_subset(val_mask, fa4_mask,
                         f"UMAP – val only, predicted {_task_name} (top-4)",
                         cfg.out_dir / "umap_split_val_fa4.png")
            log.info("  Saved umap_split*.png — compare train vs val to check cluster generalization")

        log.info("  UMAP plots saved (n=%d patches)", len(df_all))

    except ImportError:
        log.warning("  umap-learn not installed – skipping UMAP step")
        df_all = pd.read_csv(cfg.latents_csv)
        if dist_df_slim is not None:
            df_all["_dist_key"] = df_all["filename"].apply(lambda p: Path(p).name)
            df_all = df_all.merge(dist_df_slim, on="_dist_key", how="left").drop(columns=["_dist_key"])
        pred_all       = clf.predict(df_all[feat_cols].values.astype(np.float32) * feat_scale)
        pred_names_all = [id_to_label[p] for p in pred_all]

    # ------------------------------------------------------------------
    # 10. Copy patches into gt{x}pred{y} folders
    # ------------------------------------------------------------------
    log.info("Step 10: Sorting patches into gt/pred folders …")
    sort_dir = cfg.out_dir / "patch_sort"
    _sort_patches_to_folders(
        df_all, pred_names_all, label_order,
        label_col=cfg.label_col,
        sort_dir=sort_dir,
        sort_labelled=cfg.sort_labelled,
        sort_unlabelled=cfg.sort_unlabelled,
    )
    n_leaf = sum(1 for p in sort_dir.rglob("gt*") if p.is_dir())
    log.info("  Patches sorted into %d gt/pred folders under %s", n_leaf, sort_dir)

    log.info("Classification pipeline complete.  Outputs → %s", cfg.out_dir)
    return {
        "clf":        clf,
        "metrics":    metrics,
        "cv_results": cv_results,
        "df_results": df_out,
        "label_order": label_order,
    }
