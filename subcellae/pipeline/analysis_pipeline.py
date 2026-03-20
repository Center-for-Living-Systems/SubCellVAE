"""
analysis_pipeline.py
====================
Post-training analysis pipeline for SubCellAE.

Reads the ``latents.csv`` produced by :func:`run_ae_pipeline` and generates:

  1.  2-D embeddings   – UMAP (and optionally PHATE) of the latent space
  2.  Clustering       – KMeans and/or DBSCAN on the latent vectors
  3.  Scatter plots    – embedding coloured by condition, annotation, split, cluster
  4.  Latent correlation heatmap
  5.  Latent dims by condition  (box / violin)
  6.  Latent dims by annotation (box / violin, if labels present)
  7.  Class distribution bar charts
  8.  Per-class latent mean heatmap
  9.  Reconstruction MSE distribution and per-group violins
  10. Normalised MSE distribution and per-group violins

No model or dataloader is required – everything is read from the CSV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

_VALID_SPLITS = {"all", "train", "val"}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """All parameters for one analysis run.

    Parameters
    ----------
    latents_csv : Path
        Path to the ``latents.csv`` written by :func:`run_ae_pipeline`.
    out_dir : Path
        Directory where all plots and CSVs are saved.
    split_filter : str
        Which rows to analyse: ``"all"`` | ``"train"`` | ``"val"``.
    umap_methods : list[str]
        Embedding methods to run.  Currently ``"UMAP"`` and ``"PHATE"``
        are supported.
    umap_n_neighbors, umap_min_dist, umap_random_state
        UMAP hyper-parameters.
    phate_k : int
        Number of nearest neighbours for PHATE.
    kmeans_enabled, kmeans_n_clusters
        KMeans clustering settings.
    dbscan_enabled, dbscan_eps, dbscan_min_samples
        DBSCAN clustering settings.
    boxplot_kind : str
        ``"box"`` or ``"violin"`` for latent-dim plots.
    annotation_label_order : list[str] or None
        Ordered label names for annotation plots.  ``None`` → alphabetical.
    condition_name_order : list[str] or None
        Ordered condition names.  ``None`` → alphabetical.
    """

    # --- required ---
    latents_csv: Path
    out_dir: Path

    # --- data selection ---
    split_filter: str = "all"   # "all" | "train" | "val"

    # --- embedding ---
    umap_methods: list = None       # e.g. ["UMAP"] or ["UMAP", "PHATE"]
    umap_n_neighbors: int   = 15
    umap_min_dist: float    = 0.1
    umap_random_state: int  = 42
    phate_k: int            = 5

    # --- clustering ---
    kmeans_enabled: bool    = True
    kmeans_n_clusters: int  = 5
    dbscan_enabled: bool    = False
    dbscan_eps: float       = 0.5
    dbscan_min_samples: int = 10

    # --- plots ---
    boxplot_kind: str = "box"   # "box" | "violin"

    # --- label orders for plots ---
    annotation_label_order: list = None
    condition_name_order: list   = None

    def __post_init__(self):
        self.latents_csv = Path(self.latents_csv)
        self.out_dir     = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.umap_methods is None:
            self.umap_methods = ["UMAP"]

        if self.split_filter not in _VALID_SPLITS:
            raise ValueError(
                f"split_filter must be one of {_VALID_SPLITS}, "
                f"got {self.split_filter!r}"
            )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _categorical_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels,           # array-like of category values (str or int)
    order: list,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str = "tab10",
):
    """Scatter plot coloured by a categorical column."""
    palette = plt.get_cmap(cmap)
    for i, cat in enumerate(order):
        mask = np.array(labels) == cat
        ax.scatter(x[mask], y[mask],
                   label=str(cat), s=4, alpha=0.6,
                   color=palette(i / max(len(order) - 1, 1)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=3, fontsize=7, loc="best")


def _save_scatter(
    emb: np.ndarray,
    df: pd.DataFrame,
    col: str,
    order: list,
    method: str,
    title: str,
    save_path: Path,
):
    """Save a single scatter PNG coloured by *col*."""
    fig, ax = plt.subplots(figsize=(7, 6))
    _categorical_scatter(
        ax,
        emb[:, 0], emb[:, 1],
        df[col].values,
        order,
        title=title,
        xlabel=f"{method} 1",
        ylabel=f"{method} 2",
    )
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _violin_or_box(ax, data: pd.DataFrame, x: str, y: str, order: list, kind: str):
    if kind == "violin":
        sns.violinplot(data=data, x=x, y=y, order=order, ax=ax, inner="box")
    else:
        sns.boxplot(data=data, x=x, y=y, order=order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)


def _latent_by_group(
    df: pd.DataFrame,
    latent_cols: list,
    group_col: str,
    order: list,
    kind: str,
    save_path: Path,
):
    """One subplot per latent dim, coloured by *group_col*."""
    n = len(latent_cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.5, nrows * 3), sharey=False)
    axes = np.array(axes).flatten()
    melted = df[[group_col] + latent_cols].melt(
        id_vars=group_col, var_name="latent_dim", value_name="value"
    )
    for i, col in enumerate(latent_cols):
        sub = melted[melted["latent_dim"] == col]
        _violin_or_box(axes[i], sub, x=group_col, y="value", order=order, kind=kind)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Latent dims by {group_col}", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _distribution_plot(values: np.ndarray, title: str, xlabel: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values[np.isfinite(values)], bins=60, edgecolor="none", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _metric_by_group_violin(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    order: list,
    title: str,
    save_path: Path,
    kind: str = "violin",
):
    sub = df[df[metric_col].notna() & np.isfinite(df[metric_col])]
    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.5), 4))
    _violin_or_box(ax, sub, x=group_col, y=metric_col, order=order, kind=kind)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _metric_by_group_and_split(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    group_order: list,
    title: str,
    save_path: Path,
    kind: str = "box",
):
    """Box/violin plot with x=group, hue=split → 2 boxes per condition group."""
    sub = df[df[metric_col].notna() & np.isfinite(df[metric_col])].copy()
    hue_order = [s for s in ["train", "val"] if s in sub["split"].values]
    n_groups = len(group_order)
    fig, ax = plt.subplots(figsize=(max(5, n_groups * 2.5), 4))
    if kind == "violin":
        sns.violinplot(data=sub, x=group_col, y=metric_col,
                       hue="split", hue_order=hue_order,
                       order=group_order, ax=ax, inner="box", split=False)
    else:
        sns.boxplot(data=sub, x=group_col, y=metric_col,
                    hue="split", hue_order=hue_order,
                    order=group_order, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    ax.legend(title="split", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _intensity_latent_scatter(
    df: pd.DataFrame,
    latent_cols: list,
    intensity_col: str,
    save_path: Path,
):
    """Grid of scatter plots: mean_intensity vs each latent dimension."""
    n = len(latent_cols)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()

    sub = df[[intensity_col] + latent_cols].dropna()
    intensity = sub[intensity_col].values

    for i, col in enumerate(latent_cols):
        ax = axes[i]
        latent_vals = sub[col].values
        ax.scatter(intensity, latent_vals, s=0.5, alpha=0.4)
        corr = float(np.corrcoef(intensity, latent_vals)[0, 1])
        ax.set_title(f"{intensity_col} vs {col}\nr = {corr:.2f}", fontsize=9)
        ax.set_xlabel(intensity_col, fontsize=8)
        ax.set_ylabel(col, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _class_distribution(
    df: pd.DataFrame,
    col: str,
    order: list,
    title: str,
    save_path: Path,
):
    counts = df[col].value_counts()
    ordered = [c for c in order if c in counts.index]
    fig, ax = plt.subplots(figsize=(max(5, len(ordered) * 1.2), 4))
    ax.bar(range(len(ordered)), [counts.get(c, 0) for c in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _latent_mean_heatmap(
    df: pd.DataFrame,
    latent_cols: list,
    group_col: str,
    order: list,
    title: str,
    save_path: Path,
):
    rows = []
    for cat in order:
        sub = df[df[group_col] == cat][latent_cols]
        if sub.empty:
            continue
        row = {group_col: cat}
        row.update({c: sub[c].mean() for c in latent_cols})
        rows.append(row)
    if not rows:
        return
    mat = pd.DataFrame(rows).set_index(group_col)[latent_cols]
    fig, ax = plt.subplots(
        figsize=(len(latent_cols) * 1.2, len(rows) * 0.8 + 1)
    )
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("Latent dim")
    ax.set_ylabel(group_col)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def run_analysis_pipeline(cfg: AnalysisConfig):
    """Run the full post-training analysis pipeline.

    Parameters
    ----------
    cfg : AnalysisConfig
        Fully-initialised configuration object.

    Returns
    -------
    dict
        ``embeddings``, ``cluster_labels``, ``df`` (augmented DataFrame).
    """
    log.info("=" * 60)
    log.info("Analysis Pipeline")
    log.info("  latents_csv  : %s", cfg.latents_csv)
    log.info("  out_dir      : %s", cfg.out_dir)
    log.info("  split_filter : %s", cfg.split_filter)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load latents CSV
    # ------------------------------------------------------------------
    log.info("Step 1: Loading latents CSV …")
    df = pd.read_csv(cfg.latents_csv)

    if cfg.split_filter != "all":
        df = df[df["split"] == cfg.split_filter].reset_index(drop=True)
        log.info("  Filtered to split=%r  →  %d rows", cfg.split_filter, len(df))
    else:
        log.info("  Loaded %d rows", len(df))

    latent_cols = [c for c in df.columns if c.startswith("z_")]
    latents     = df[latent_cols].values.astype(np.float32)
    log.info("  Latent dims: %d  (%s … %s)", len(latent_cols),
             latent_cols[0], latent_cols[-1])

    # -- resolve label orders -----------------------------------------
    cond_order = cfg.condition_name_order or sorted(df["condition_name"].dropna().unique().tolist())
    split_order = ["train", "val"]

    has_annotation = (
        "annotation_label_name" in df.columns
        and df["annotation_label_name"].notna().any()
        and (df["annotation_label"] != -1).any()
    )
    if has_annotation:
        ann_vals = df.loc[df["annotation_label"] != -1, "annotation_label_name"]
        ann_order = cfg.annotation_label_order or sorted(ann_vals.dropna().unique().tolist())
    else:
        ann_order = []
    log.info("  Conditions: %s", cond_order)
    log.info("  Annotations present: %s%s",
             has_annotation, f"  {ann_order}" if has_annotation else "")

    results = {}

    # ------------------------------------------------------------------
    # 2. 2-D Embeddings
    # ------------------------------------------------------------------
    log.info("Step 2: Computing embeddings …")
    embeddings: dict[str, np.ndarray] = {}

    for method in cfg.umap_methods:
        log.info("  %s …", method)
        if method.upper() == "UMAP":
            from umap import UMAP
            reducer = UMAP(
                n_components=2,
                n_neighbors=cfg.umap_n_neighbors,
                min_dist=cfg.umap_min_dist,
                random_state=cfg.umap_random_state,
            )
            emb = reducer.fit_transform(latents)
            joblib.dump(reducer, str(cfg.out_dir / "umap_model.pkl"))

        elif method.upper() == "PHATE":
            try:
                import phate
            except ImportError:
                log.warning("  phate package not installed – skipping PHATE")
                continue
            reducer = phate.PHATE(k=cfg.phate_k, random_state=42)
            emb = reducer.fit_transform(latents)
            joblib.dump(reducer, str(cfg.out_dir / "phate_model.pkl"))

        else:
            log.warning("  Unknown embedding method %r – skipping", method)
            continue

        embeddings[method] = emb
        df[f"{method}_1"] = emb[:, 0]
        df[f"{method}_2"] = emb[:, 1]

    results["embeddings"] = embeddings

    # ------------------------------------------------------------------
    # 3. Clustering
    # ------------------------------------------------------------------
    log.info("Step 3: Clustering …")
    from subcellae.clustering.clustering import kmeans_cluster, DBSCAN_cluster

    cluster_labels: dict = {}

    if cfg.kmeans_enabled:
        log.info("  KMeans (k=%d) …", cfg.kmeans_n_clusters)
        _, km_labels = kmeans_cluster(
            latents, cfg.kmeans_n_clusters,
            str(cfg.out_dir), "kmeans_model",
        )
        cluster_labels["kmeans"] = km_labels
        df["kmeans_cluster"] = km_labels.astype(str)

    if cfg.dbscan_enabled:
        log.info("  DBSCAN (eps=%g, min_samples=%d) …",
                 cfg.dbscan_eps, cfg.dbscan_min_samples)
        _, db_labels = DBSCAN_cluster(
            latents, cfg.dbscan_eps, cfg.dbscan_min_samples,
            str(cfg.out_dir), "dbscan_model",
        )
        cluster_labels["dbscan"] = db_labels
        df["dbscan_cluster"] = db_labels.astype(str)

    results["cluster_labels"] = cluster_labels

    # ------------------------------------------------------------------
    # 4. Scatter plots
    # ------------------------------------------------------------------
    log.info("Step 4: Scatter plots …")

    for method, emb in embeddings.items():
        mdir = cfg.out_dir / f"{method.lower()}"
        mdir.mkdir(exist_ok=True)

        # — condition
        _save_scatter(emb, df, "condition_name", cond_order, method,
                      f"{method} – condition",
                      mdir / "by_condition.png")

        # — split
        _save_scatter(emb, df, "split", split_order, method,
                      f"{method} – split",
                      mdir / "by_split.png")

        # — annotation (if present)
        if has_annotation:
            df_ann = df[df["annotation_label"] != -1]
            emb_ann = emb[df["annotation_label"].values != -1]
            _save_scatter(emb_ann, df_ann, "annotation_label_name", ann_order,
                          method, f"{method} – annotation label",
                          mdir / "by_annotation.png")

        # — kmeans cluster
        if "kmeans_cluster" in df.columns:
            km_order = sorted(df["kmeans_cluster"].unique().tolist())
            _save_scatter(emb, df, "kmeans_cluster", km_order, method,
                          f"{method} – KMeans (k={cfg.kmeans_n_clusters})",
                          mdir / "by_kmeans.png")

        # — dbscan cluster
        if "dbscan_cluster" in df.columns:
            db_order = sorted(df["dbscan_cluster"].unique().tolist())
            _save_scatter(emb, df, "dbscan_cluster", db_order, method,
                          f"{method} – DBSCAN",
                          mdir / "by_dbscan.png")

    # ------------------------------------------------------------------
    # 5. Latent correlation heatmap
    # ------------------------------------------------------------------
    log.info("Step 5: Latent correlation heatmap …")
    corr = df[latent_cols].corr()
    corr.to_csv(str(cfg.out_dir / "latent_correlation.csv"))
    fig, ax = plt.subplots(figsize=(len(latent_cols) * 1.0 + 1, len(latent_cols) * 0.9 + 1))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Latent dimension correlation")
    fig.tight_layout()
    fig.savefig(str(cfg.out_dir / "latent_correlation.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Latent dims by group
    # ------------------------------------------------------------------
    log.info("Step 6: Latent dims by group (box/violin) …")
    _latent_by_group(
        df, latent_cols,
        group_col="condition_name", order=cond_order,
        kind=cfg.boxplot_kind,
        save_path=cfg.out_dir / "latent_by_condition.png",
    )
    if has_annotation:
        df_ann = df[df["annotation_label"] != -1]
        _latent_by_group(
            df_ann, latent_cols,
            group_col="annotation_label_name", order=ann_order,
            kind=cfg.boxplot_kind,
            save_path=cfg.out_dir / "latent_by_annotation.png",
        )

    # ------------------------------------------------------------------
    # 7. Class distribution bar charts
    # ------------------------------------------------------------------
    log.info("Step 7: Class distribution …")
    _class_distribution(df, "condition_name", cond_order,
                        "Patch count by condition",
                        cfg.out_dir / "distribution_condition.png")
    if has_annotation:
        df_ann = df[df["annotation_label"] != -1]
        _class_distribution(df_ann, "annotation_label_name", ann_order,
                            "Patch count by annotation label",
                            cfg.out_dir / "distribution_annotation.png")

    # ------------------------------------------------------------------
    # 8. Per-class latent mean heatmaps
    # ------------------------------------------------------------------
    log.info("Step 8: Per-class latent mean heatmaps …")
    _latent_mean_heatmap(
        df, latent_cols,
        group_col="condition_name", order=cond_order,
        title="Mean latent value per condition",
        save_path=cfg.out_dir / "latent_mean_by_condition.png",
    )
    if has_annotation:
        df_ann = df[df["annotation_label"] != -1]
        _latent_mean_heatmap(
            df_ann, latent_cols,
            group_col="annotation_label_name", order=ann_order,
            title="Mean latent value per annotation label",
            save_path=cfg.out_dir / "latent_mean_by_annotation.png",
        )

    # ------------------------------------------------------------------
    # 9. Reconstruction MSE
    # ------------------------------------------------------------------
    log.info("Step 9: Reconstruction MSE plots …")
    if "recon_mse" in df.columns:
        _distribution_plot(
            df["recon_mse"].values,
            title="Reconstruction MSE – all patches",
            xlabel="Reconstruction MSE",
            save_path=cfg.out_dir / "mse_distribution.png",
        )
        _metric_by_group_and_split(
            df, "recon_mse", "condition_name", cond_order,
            title="Reconstruction MSE by condition × split",
            save_path=cfg.out_dir / "mse_by_condition_split.png",
            kind=cfg.boxplot_kind,
        )
        if has_annotation:
            df_ann = df[df["annotation_label"] != -1]
            _metric_by_group_and_split(
                df_ann, "recon_mse", "annotation_label_name", ann_order,
                title="Reconstruction MSE by annotation label × split",
                save_path=cfg.out_dir / "mse_by_annotation_split.png",
                kind=cfg.boxplot_kind,
            )
    else:
        log.warning("  'recon_mse' column not found – skipping MSE plots")

    # ------------------------------------------------------------------
    # 10. Normalised MSE
    # ------------------------------------------------------------------
    log.info("Step 10: Normalised MSE plots …")
    if "norm_mse" in df.columns:
        _distribution_plot(
            df["norm_mse"].values,
            title="Normalised MSE – all patches",
            xlabel="norm_mse  (MSE / mean intensity)",
            save_path=cfg.out_dir / "norm_mse_distribution.png",
        )
        _metric_by_group_and_split(
            df, "norm_mse", "condition_name", cond_order,
            title="Normalised MSE by condition × split",
            save_path=cfg.out_dir / "norm_mse_by_condition_split.png",
            kind=cfg.boxplot_kind,
        )
        if has_annotation:
            df_ann = df[df["annotation_label"] != -1]
            _metric_by_group_and_split(
                df_ann, "norm_mse", "annotation_label_name", ann_order,
                title="Normalised MSE by annotation label × split",
                save_path=cfg.out_dir / "norm_mse_by_annotation_split.png",
                kind=cfg.boxplot_kind,
            )
    else:
        log.warning("  'norm_mse' column not found – skipping norm_mse plots")

    # ------------------------------------------------------------------
    # 11. Mean intensity vs latent dimension scatter
    # ------------------------------------------------------------------
    log.info("Step 11: Mean intensity vs latent dimension scatter …")
    if "mean_intensity" in df.columns:
        _intensity_latent_scatter(
            df, latent_cols,
            intensity_col="mean_intensity",
            save_path=cfg.out_dir / "intensity_vs_latent.png",
        )
    else:
        log.warning("  'mean_intensity' column not found – skipping intensity scatter")

    # ------------------------------------------------------------------
    # Save augmented CSV
    # ------------------------------------------------------------------
    out_csv = cfg.out_dir / "analysis_results.csv"
    df.to_csv(str(out_csv), index=False)
    log.info("Augmented CSV saved → %s  (%d rows)", out_csv, len(df))

    results["df"] = df
    log.info("Analysis pipeline complete.  Outputs → %s", cfg.out_dir)
    return results
