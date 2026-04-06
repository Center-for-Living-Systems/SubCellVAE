"""
cls_apply_pipeline.py
=====================
Classifier application pipeline for unlabelled data.

Loads a trained ``lgbm_model.pkl`` (plus optional ``umap_all_model.pkl``) from
a previous :func:`run_classification_pipeline` run, applies it to the latent
features in a ``latents_newdata.csv`` (produced by :func:`run_ae_apply_pipeline`),
and saves:

  - ``predictions_all.csv``     – all patches + predicted label + per-class probas
  - ``umap_pred.png``           – UMAP coloured by predicted label
  - ``umap_condition.png``      – UMAP coloured by condition
  - ``pred_distribution.png``   – bar chart of predicted-label counts

No ground-truth labels are required or used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClsApplyConfig:
    """All parameters for one classifier-apply run.

    Parameters
    ----------
    latents_csv : Path
        ``latents_newdata.csv`` from :func:`run_ae_apply_pipeline`.
    model_pkl : Path
        ``lgbm_model.pkl`` from a previous classification pipeline run.
    out_dir : Path
        Directory where all outputs are saved.
    label_order : list[str] or None
        Ordered class names for the predicted label.  If None, inferred from
        the LightGBM model's ``classes_`` attribute.
    umap_model_pkl : Path or None
        Optional path to ``umap_all_model.pkl`` from the classification pipeline.
        When provided, the existing UMAP transform is used; otherwise a new UMAP
        is fit on the new data.
    feature_cols : list[str] or None
        Explicit feature column names.  None → auto-detect all ``z_*`` columns.
    dist_patch_prep_dirs : list[str]
        Patch-prep output directories for distance features (same semantics as
        in :class:`ClassificationConfig`).  Leave empty for latent-only features.
    dist_feature_weight : float
        Multiplicative weight for distance features (default 100.0).
    umap_n_neighbors : int
    umap_min_dist : float
    umap_random_state : int
    """

    # --- required ---
    latents_csv: Path
    model_pkl: Path
    out_dir: Path

    # --- optional ---
    label_order: list = None
    umap_model_pkl: Path = None

    # --- features ---
    feature_cols: list = None
    dist_patch_prep_dirs: list = None
    dist_feature_weight: float = 100.0

    # --- UMAP (used only when umap_model_pkl is None) ---
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_random_state: int = 42

    def __post_init__(self):
        self.latents_csv = Path(self.latents_csv)
        self.model_pkl   = Path(self.model_pkl)
        self.out_dir     = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self.umap_model_pkl:
            self.umap_model_pkl = Path(self.umap_model_pkl)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dist_csvs(dirs: list) -> pd.DataFrame:
    """Load and concatenate patch-prep record CSVs (mirrors classification_pipeline)."""
    dfs = []
    for d in dirs:
        d = Path(d)
        candidates = list(d.glob("data_prep_record_*_to_*.csv"))
        if not candidates:
            log.warning("  No data_prep_record_*_to_*.csv in %s – skipping", d)
            continue

        def _end_idx(p: Path) -> int:
            last = p.stem.rsplit("_to_", 1)[-1]
            try:
                return int(last)
            except ValueError:
                return -1

        best = max(candidates, key=_end_idx)
        log.info("  Dist CSV: %s", best)
        dfs.append(pd.read_csv(best))

    if not dfs:
        raise FileNotFoundError(
            f"No data_prep_record CSVs found in: {dirs}"
        )
    return pd.concat(dfs, ignore_index=True)


def _plot_umap(emb: np.ndarray, labels, palette_name: str,
               title: str, save_path: Path, label_order=None) -> None:
    """Scatter UMAP coloured by *labels* (str or numeric)."""
    unique = list(label_order) if label_order else sorted(set(str(l) for l in labels))
    palette = plt.get_cmap("tab10")
    color_map = {lbl: palette(i % 10) for i, lbl in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique:
        mask = np.array([str(l) == lbl for l in labels])
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=4, alpha=0.6, color=color_map[lbl], label=str(lbl))
    ax.legend(markerscale=3, loc="best", fontsize=8, title=title, title_fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_bar(counts: pd.Series, title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def run_cls_apply_pipeline(cfg: ClsApplyConfig) -> pd.DataFrame:
    """Apply a trained LightGBM classifier to an unlabelled latents CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame of all patches with added prediction columns.
    """
    log.info("=" * 60)
    log.info("Classifier Apply Pipeline")
    log.info("  latents_csv : %s", cfg.latents_csv)
    log.info("  model_pkl   : %s", cfg.model_pkl)
    log.info("  out_dir     : %s", cfg.out_dir)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load latents CSV
    # ------------------------------------------------------------------
    log.info("Step 1: Loading latents …")
    df = pd.read_csv(cfg.latents_csv)
    log.info("  %d patches loaded", len(df))

    # ------------------------------------------------------------------
    # 2. Optionally merge distance features
    # ------------------------------------------------------------------
    if cfg.dist_patch_prep_dirs:
        log.info("Step 2: Merging distance features …")
        dist_df = _load_dist_csvs(cfg.dist_patch_prep_dirs)
        # Normalise join key (crop_img_filename → filename without path)
        dist_df["_join_key"] = dist_df["crop_img_filename"].apply(
            lambda x: Path(x).name
        )
        df["_join_key"] = df["filename"].apply(lambda x: Path(x).name)
        dist_cols = [c for c in dist_df.columns
                     if c.startswith("d") and c[1:].isdigit()
                     or c == "equiv_diam"]
        df = df.merge(dist_df[["_join_key"] + dist_cols],
                      on="_join_key", how="left")
        df.drop(columns=["_join_key"], inplace=True)
        log.info("  Distance columns merged: %s", dist_cols)

    # ------------------------------------------------------------------
    # 3. Build feature matrix
    # ------------------------------------------------------------------
    log.info("Step 3: Building feature matrix …")
    if cfg.feature_cols:
        feat_cols = list(cfg.feature_cols)
    else:
        feat_cols = sorted([c for c in df.columns if c.startswith("z_")])

    if cfg.dist_patch_prep_dirs:
        # Match the feature selection used during training: d\d{2} columns only
        # (equiv_diam is merged but excluded from the feature matrix, same as
        # classification_pipeline.py line: d_feats = [c for c in dist_cols if c.startswith("d")])
        import re as _re
        dist_feat_cols = [c for c in df.columns if _re.fullmatch(r"d\d{2}", c)]
        feat_cols = feat_cols + dist_feat_cols

    X = df[feat_cols].values.astype(np.float32)

    # Apply distance feature weight
    if cfg.dist_patch_prep_dirs and cfg.dist_feature_weight != 1.0:
        n_lat = len([c for c in feat_cols if c.startswith("z_")])
        X[:, n_lat:] *= cfg.dist_feature_weight

    log.info("  Feature columns: %d  (%d latent + %d dist)",
             len(feat_cols),
             len([c for c in feat_cols if c.startswith("z_")]),
             len(feat_cols) - len([c for c in feat_cols if c.startswith("z_")]))

    # ------------------------------------------------------------------
    # 4. Load classifier and predict
    # ------------------------------------------------------------------
    log.info("Step 4: Loading classifier and predicting …")
    clf = joblib.load(str(cfg.model_pkl))
    pred_labels  = clf.predict(X)
    pred_probas  = clf.predict_proba(X)

    # clf.classes_ may be integer indices [0,1,2,...]; pred_labels are numpy
    # integers.  Always map to string class names so downstream plots work.
    if cfg.label_order:
        classes = list(cfg.label_order)
        df["pred_label"] = [classes[int(i)] for i in pred_labels]
    else:
        classes = [str(c) for c in clf.classes_]
        df["pred_label"] = [str(i) for i in pred_labels]
    log.info("  Classes: %s", classes)

    df["max_prob"] = pred_probas.max(axis=1)
    for j, cls_name in enumerate(clf.classes_):
        df[f"proba_{cls_name}"] = pred_probas[:, j]

    # ------------------------------------------------------------------
    # 5. Save predictions CSV
    # ------------------------------------------------------------------
    out_csv = cfg.out_dir / "predictions_all.csv"
    df.to_csv(out_csv, index=False)
    log.info("Predictions saved → %s", out_csv)

    # ------------------------------------------------------------------
    # 6. UMAP
    # ------------------------------------------------------------------
    log.info("Step 5: UMAP embedding …")
    try:
        import umap as umap_lib

        if cfg.umap_model_pkl and Path(cfg.umap_model_pkl).exists():
            log.info("  Loading existing UMAP model from %s", cfg.umap_model_pkl)
            umap_model = joblib.load(str(cfg.umap_model_pkl))
            emb = umap_model.transform(X)
        else:
            log.info("  Fitting new UMAP (n_neighbors=%d, min_dist=%.2f)",
                     cfg.umap_n_neighbors, cfg.umap_min_dist)
            umap_model = umap_lib.UMAP(
                n_components=2,
                n_neighbors=cfg.umap_n_neighbors,
                min_dist=cfg.umap_min_dist,
                random_state=cfg.umap_random_state,
            )
            emb = umap_model.fit_transform(X)
            joblib.dump(umap_model, str(cfg.out_dir / "umap_newdata_model.pkl"))

        # Save embedding coords
        emb_df = df[["filename"]].copy()
        emb_df["umap_1"] = emb[:, 0]
        emb_df["umap_2"] = emb[:, 1]
        emb_df.to_csv(cfg.out_dir / "umap_embedding.csv", index=False)

        # UMAP by predicted label
        _plot_umap(
            emb, df["pred_label"].tolist(),
            palette_name="tab10",
            title="Predicted label",
            save_path=cfg.out_dir / "umap_pred.png",
            label_order=classes,
        )

        # UMAP by condition
        if "condition_name" in df.columns:
            _plot_umap(
                emb, df["condition_name"].tolist(),
                palette_name="tab10",
                title="Condition",
                save_path=cfg.out_dir / "umap_condition.png",
            )

    except ImportError:
        log.warning("  umap-learn not available; skipping UMAP plots.")

    # ------------------------------------------------------------------
    # 7. Predicted-label distribution bar chart
    # ------------------------------------------------------------------
    counts = df["pred_label"].value_counts().reindex(classes, fill_value=0)
    _plot_bar(counts, "Predicted label distribution",
              cfg.out_dir / "pred_distribution.png")
    log.info("Distribution plot saved.")

    log.info("Classifier Apply Pipeline complete.")
    return df
