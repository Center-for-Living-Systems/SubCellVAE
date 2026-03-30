"""
run_cross_classification_vis.py
================================
Post-classification visualization pipeline.

Merges predictions_all.csv from the FA-type and Position classifiers,
loads latent-space UMAP coordinates, and saves:

UMAP plots (per condition / split subset):
  umap_{subset}_fa_type.png         – coloured by FA-type prediction
  umap_{subset}_position.png        – coloured by Position prediction
  umap_labelled_true_fa_type.png    – labelled patches, true FA-type
  umap_labelled_true_position.png   – labelled patches, true Position

Crosstab heatmaps  (Position rows × FA-type columns):
  crosstab_pred_counts_{subset}.png       – prediction × prediction, counts
  crosstab_pred_norm_row_{subset}.png     – row-normalised  (given Position → FA-type)
  crosstab_pred_norm_col_{subset}.png     – col-normalised  (given FA-type → Position)
  crosstab_true_both_{subset}.png         – true × true (labelled only)
  crosstab_truepos_predfa_{subset}.png    – true Position × predicted FA-type
  crosstab_predpos_truefa_{subset}.png    – predicted Position × true FA-type

  (subsets: all, control, ycomp, train, val)

Merged CSV:
  cross_classification_results.csv

Usage
-----
    python scripts/run_cross_classification_vis.py config/config_cross_classification_vis.yaml
    python scripts/run_cross_classification_vis.py config/config_cross_classification_vis.yaml --dry_run
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
})
import joblib

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.utils.label_colors import (
    classification_label_order,
    classification_label_to_color,
    position_label_order,
    position_label_to_color,
)
from subcellae.utils.config_utils import resolve_root

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CrossVisConfig:
    latents_csv          : Path
    fa_type_results_csv  : Path
    position_results_csv : Path
    out_dir              : Path
    umap_model_pkl       : Path | None = None  # None → fit new UMAP
    fa_type_label_col    : str  = "classification"   # true-label col in FA results
    position_label_col   : str  = "Position"         # true-label col in Position results
    fa_type_order        : list = field(default_factory=lambda: list(classification_label_order[:-1]))
    position_order       : list = field(default_factory=lambda: list(position_label_order))
    random_state         : int  = 42
    log_level            : str  = "INFO"


def load_config(yaml_path: str | Path, root_folder: str | None = None) -> CrossVisConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section, key, default=None):
        return raw.get(section, {}).get(key, default)

    umap_pkl = _get("input", "umap_model_pkl", None)
    return CrossVisConfig(
        latents_csv          = Path(str(_get("input",  "latents_csv",          ""))),
        fa_type_results_csv  = Path(str(_get("input",  "fa_type_results_csv",  ""))),
        position_results_csv = Path(str(_get("input",  "position_results_csv", ""))),
        out_dir              = Path(str(_get("output", "out_dir", "results/cross_classification_vis"))),
        umap_model_pkl       = Path(str(umap_pkl)) if umap_pkl else None,
        fa_type_label_col    = str(_get("labels", "fa_type_label_col", "classification")),
        position_label_col   = str(_get("labels", "position_label_col", "Position")),
        fa_type_order        = _get("labels", "fa_type_order",
                                    list(classification_label_order[:-1])),
        position_order       = _get("labels", "position_order",
                                    list(position_label_order)),
        random_state         = int(_get("misc", "random_state", 42)),
        log_level            = str(_get("misc", "log_level", "INFO")),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_umap(
    emb: np.ndarray,
    labels: np.ndarray,
    label_order: list,
    label_to_color: dict | None,
    title: str,
    save_path: Path,
    xlim=None,
    ylim=None,
):
    """Scatter plot of a 2-D UMAP embedding coloured by label."""
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, lbl in enumerate(label_order):
        mask = labels == lbl
        if not mask.any():
            continue
        color = (label_to_color.get(lbl) if label_to_color else None) \
                or palette(i / max(len(label_order) - 1, 1))
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   label=lbl, s=4, alpha=0.6, color=color)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(markerscale=3, loc="best")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", save_path.name)


def _plot_umap_combined(
    emb: np.ndarray,
    fa_labels: np.ndarray,
    pos_labels: np.ndarray,
    fa_order: list,
    pos_order: list,
    title: str,
    save_path: Path,
    xlim=None,
    ylim=None,
):
    """Scatter plot of a 2-D UMAP embedding coloured by (FA-type, Position) pair.

    20 pairs (5 FA types × 4 positions) are encoded with tab10 (10 colours) ×
    2 markers ('o' and '*').  Pair index 0-9 → marker 'o'; 10-19 → marker '*'.
    """
    palette = plt.get_cmap("tab10")
    markers = ["o", "*"]
    marker_sizes = {"o": 4, "*": 6}

    # Build ordered list of all pairs and assign colour + marker
    pairs = [(fa, pos) for fa in fa_order for pos in pos_order]  # 20 pairs
    pair_style = {}
    for idx, pair in enumerate(pairs):
        pair_style[pair] = {
            "color":  palette(idx % 10),
            "marker": markers[idx // 10],
        }

    fig, ax = plt.subplots(figsize=(10, 8))
    for pair in pairs:
        fa, pos = pair
        mask = (fa_labels == fa) & (pos_labels == pos)
        if not mask.any():
            continue
        style = pair_style[pair]
        m = style["marker"]
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            label=f"{fa} | {pos}",
            s=marker_sizes[m],
            alpha=0.6,
            color=style["color"],
            marker=m,
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(markerscale=3, loc="best", fontsize=8, ncol=1,
              title="FA type | Position", title_fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", save_path.name)


def _plot_crosstab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    row_order: list,
    col_order: list,
    title: str,
    save_path: Path,
    normalize: str | None = None,  # None=counts  "index"=row-norm  "columns"=col-norm
):
    """Heatmap of a crosstab between two categorical columns."""
    sub = df[[row_col, col_col]].dropna()
    if sub.empty:
        log.warning("  Skipping %s – no overlapping data", save_path.name)
        return

    ct = pd.crosstab(sub[row_col], sub[col_col])
    ct = ct.reindex(index=row_order, columns=col_order, fill_value=0)

    if normalize == "index":
        row_sums = ct.sum(axis=1).replace(0, np.nan)
        values   = ct.divide(row_sums, axis=0).fillna(0).values
        fmt_str, cbar_label, cmap = ".2f", "Row proportion", "Blues"
        vmin, vmax = 0.0, 1.0
    elif normalize == "columns":
        col_sums = ct.sum(axis=0).replace(0, np.nan)
        values   = ct.divide(col_sums, axis=1).fillna(0).values
        fmt_str, cbar_label, cmap = ".2f", "Column proportion", "Oranges"
        vmin, vmax = 0.0, 1.0
    else:
        values   = ct.values.astype(float)
        fmt_str, cbar_label, cmap = "d", "Count", "Greens"
        vmin, vmax = 0.0, float(values.max()) if values.max() > 0 else 1.0

    n_rows, n_cols = len(row_order), len(col_order)
    fig_w = max(7, n_cols * 1.8 + 1.5)
    fig_h = max(5, n_rows * 1.1 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticklabels(row_order)
    ax.set_xlabel(col_col.replace("_", " ").title())
    ax.set_ylabel(row_col.replace("_", " ").title())
    ax.set_title(title)

    # Annotate each cell
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            txt = format(int(round(val)), "d") if fmt_str == "d" else format(val, fmt_str)
            text_color = "white" if val > vmax * 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=text_color)

    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    log.info("  Saved: %s", save_path.name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cross_vis(cfg: CrossVisConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", cfg.out_dir)

    # ------------------------------------------------------------------
    # 1. Load and merge prediction CSVs from both classifiers
    # ------------------------------------------------------------------
    log.info("Step 1: Loading classifier predictions …")

    def _norm_key(s: str) -> str:
        return Path(str(s)).name

    df_fa = pd.read_csv(cfg.fa_type_results_csv)
    df_fa["_key"] = df_fa["filename"].apply(_norm_key)
    fa_rename = {"pred_label": "pred_fa_type"}
    if cfg.fa_type_label_col in df_fa.columns:
        fa_rename[cfg.fa_type_label_col] = "true_fa_type"
    df_fa = df_fa.rename(columns=fa_rename)
    fa_keep = ["_key", "filename", "condition_name", "group", "split",
               "pred_fa_type", "max_prob"] + \
              (["true_fa_type"] if "true_fa_type" in df_fa.columns else [])
    df_fa = df_fa[[c for c in fa_keep if c in df_fa.columns]]
    df_fa = df_fa.rename(columns={"max_prob": "max_prob_fa_type"})
    log.info("  FA-type:  %d rows", len(df_fa))

    df_pos = pd.read_csv(cfg.position_results_csv)
    df_pos["_key"] = df_pos["filename"].apply(_norm_key)
    pos_rename = {"pred_label": "pred_position"}
    if cfg.position_label_col in df_pos.columns:
        pos_rename[cfg.position_label_col] = "true_position"
    df_pos = df_pos.rename(columns=pos_rename)
    pos_keep = ["_key", "pred_position", "max_prob"] + \
               (["true_position"] if "true_position" in df_pos.columns else [])
    df_pos = df_pos[[c for c in pos_keep if c in df_pos.columns]]
    df_pos = df_pos.rename(columns={"max_prob": "max_prob_position"})
    log.info("  Position: %d rows", len(df_pos))

    df = df_fa.merge(df_pos, on="_key", how="inner")
    log.info("  Merged:   %d rows (inner join on filename)", len(df))

    # ------------------------------------------------------------------
    # 2. Load latents and build / load UMAP embedding
    # ------------------------------------------------------------------
    log.info("Step 2: Building UMAP embedding …")

    df_lat = pd.read_csv(cfg.latents_csv)
    df_lat["_key"] = df_lat["filename"].apply(_norm_key)
    feat_cols = [c for c in df_lat.columns if c.startswith("z_")]
    X_all = df_lat[feat_cols].values.astype(np.float32)

    if cfg.umap_model_pkl and cfg.umap_model_pkl.exists():
        log.info("  Loading UMAP model: %s", cfg.umap_model_pkl)
        reducer = joblib.load(str(cfg.umap_model_pkl))
        emb_all = reducer.transform(X_all)
    else:
        log.info("  Fitting new UMAP on %d × %d …", *X_all.shape)
        try:
            from umap import UMAP as _UMAP
        except ImportError:
            raise ImportError("umap-learn is required: pip install umap-learn")
        reducer = _UMAP(n_components=2, random_state=cfg.random_state)
        emb_all = reducer.fit_transform(X_all)
        joblib.dump(reducer, str(cfg.out_dir / "umap_model.pkl"))
        log.info("  UMAP model saved.")

    # Attach 2-D coordinates to each patch in the merged dataframe
    df_emb = pd.DataFrame({
        "_key":  df_lat["_key"].values,
        "umap1": emb_all[:, 0],
        "umap2": emb_all[:, 1],
    })
    df = df.merge(df_emb, on="_key", how="left").drop(columns=["_key"])
    log.info("  UMAP coords attached for %d / %d rows",
             df["umap1"].notna().sum(), len(df))

    # Shared axis limits from the full latent UMAP (5 % padding)
    _pad = 0.05
    u1_range = emb_all[:, 0].max() - emb_all[:, 0].min()
    u2_range = emb_all[:, 1].max() - emb_all[:, 1].min()
    xlim = (emb_all[:, 0].min() - _pad * u1_range,
            emb_all[:, 0].max() + _pad * u1_range)
    ylim = (emb_all[:, 1].min() - _pad * u2_range,
            emb_all[:, 1].max() + _pad * u2_range)

    # ------------------------------------------------------------------
    # 3. UMAP plots
    # ------------------------------------------------------------------
    log.info("Step 3: Saving UMAP plots …")

    cond_col = "condition_name" if "condition_name" in df.columns else None
    spl_col  = "split"          if "split"          in df.columns else None

    all_mask   = np.ones(len(df), dtype=bool)
    ctrl_mask  = (df[cond_col] == "control").values if cond_col else all_mask
    ycomp_mask = (df[cond_col] == "ycomp").values   if cond_col else all_mask
    train_mask = (df[spl_col]  == "train").values   if spl_col  else all_mask
    val_mask   = (df[spl_col]  == "val").values     if spl_col  else all_mask
    umap_valid = df["umap1"].notna().values

    subsets = {
        "all":     all_mask,
        "control": ctrl_mask,
        "ycomp":   ycomp_mask,
        "train":   train_mask,
        "val":     val_mask,
    }

    for subset_name, smask in subsets.items():
        mask = smask & umap_valid
        if not mask.any():
            log.warning("  Skipping UMAP subset '%s' – no rows with embedding", subset_name)
            continue
        emb_sub = np.column_stack([df.loc[mask, "umap1"], df.loc[mask, "umap2"]])

        _plot_umap(
            emb_sub,
            df.loc[mask, "pred_fa_type"].values,
            cfg.fa_type_order,
            classification_label_to_color,
            title=f"UMAP – {subset_name}, predicted FA type",
            save_path=cfg.out_dir / f"umap_{subset_name}_fa_type.png",
            xlim=xlim, ylim=ylim,
        )
        _plot_umap(
            emb_sub,
            df.loc[mask, "pred_position"].values,
            cfg.position_order,
            position_label_to_color,
            title=f"UMAP – {subset_name}, predicted Position",
            save_path=cfg.out_dir / f"umap_{subset_name}_position.png",
            xlim=xlim, ylim=ylim,
        )

    # Combined (FA-type × Position) UMAP plots – tab10 colour × o/* marker
    for subset_name, smask in subsets.items():
        mask = smask & umap_valid
        if not mask.any():
            continue
        emb_sub = np.column_stack([df.loc[mask, "umap1"], df.loc[mask, "umap2"]])

        _plot_umap_combined(
            emb_sub,
            df.loc[mask, "pred_fa_type"].values,
            df.loc[mask, "pred_position"].values,
            cfg.fa_type_order,
            cfg.position_order,
            title=f"UMAP – {subset_name}, predicted FA type × Position",
            save_path=cfg.out_dir / f"umap_{subset_name}_combined_pred.png",
            xlim=xlim, ylim=ylim,
        )

    if "true_fa_type" in df.columns and "true_position" in df.columns:
        mask = (df["true_fa_type"].notna() & df["true_position"].notna()).values & umap_valid
        if mask.any():
            emb_sub = np.column_stack([df.loc[mask, "umap1"], df.loc[mask, "umap2"]])
            _plot_umap_combined(
                emb_sub,
                df.loc[mask, "true_fa_type"].values,
                df.loc[mask, "true_position"].values,
                cfg.fa_type_order,
                cfg.position_order,
                title="UMAP – labelled patches, true FA type × Position",
                save_path=cfg.out_dir / "umap_labelled_combined_true.png",
                xlim=xlim, ylim=ylim,
            )

    # True-label UMAP (labelled patches only)
    if "true_fa_type" in df.columns:
        mask = df["true_fa_type"].notna().values & umap_valid
        if mask.any():
            emb_sub = np.column_stack([df.loc[mask, "umap1"], df.loc[mask, "umap2"]])
            _plot_umap(emb_sub, df.loc[mask, "true_fa_type"].values,
                       cfg.fa_type_order, classification_label_to_color,
                       "UMAP – labelled patches, true FA type",
                       cfg.out_dir / "umap_labelled_true_fa_type.png",
                       xlim=xlim, ylim=ylim)

    if "true_position" in df.columns:
        mask = df["true_position"].notna().values & umap_valid
        if mask.any():
            emb_sub = np.column_stack([df.loc[mask, "umap1"], df.loc[mask, "umap2"]])
            _plot_umap(emb_sub, df.loc[mask, "true_position"].values,
                       cfg.position_order, position_label_to_color,
                       "UMAP – labelled patches, true Position",
                       cfg.out_dir / "umap_labelled_true_position.png",
                       xlim=xlim, ylim=ylim)

    # ------------------------------------------------------------------
    # 4. Crosstab plots  (Position rows × FA-type columns)
    # ------------------------------------------------------------------
    log.info("Step 4: Saving crosstab plots …")

    subsets_ct = {
        "all":     np.ones(len(df), dtype=bool),
        "control": ctrl_mask,
        "ycomp":   ycomp_mask,
        "train":   train_mask,
        "val":     val_mask,
    }

    has_true_fa  = "true_fa_type"  in df.columns
    has_true_pos = "true_position" in df.columns

    for subset_name, smask in subsets_ct.items():
        if not smask.any():
            continue
        sub = df[smask]

        # ── predicted × predicted ─────────────────────────────────────────
        _plot_crosstab(
            sub, "pred_position", "pred_fa_type",
            cfg.position_order, cfg.fa_type_order,
            f"Position vs FA-type predictions – {subset_name} (counts)",
            cfg.out_dir / f"crosstab_pred_counts_{subset_name}.png",
        )
        _plot_crosstab(
            sub, "pred_position", "pred_fa_type",
            cfg.position_order, cfg.fa_type_order,
            f"Position vs FA-type predictions – {subset_name} (row-normalised)",
            cfg.out_dir / f"crosstab_pred_norm_row_{subset_name}.png",
            normalize="index",
        )
        _plot_crosstab(
            sub, "pred_position", "pred_fa_type",
            cfg.position_order, cfg.fa_type_order,
            f"Position vs FA-type predictions – {subset_name} (col-normalised)",
            cfg.out_dir / f"crosstab_pred_norm_col_{subset_name}.png",
            normalize="columns",
        )

        # ── true × true and mixed (labelled patches only) ─────────────────
        if has_true_fa and has_true_pos:
            lbl_mask = sub["true_fa_type"].notna() & sub["true_position"].notna()
            lbl = sub[lbl_mask]
            if not lbl.empty:
                _plot_crosstab(
                    lbl, "true_position", "true_fa_type",
                    cfg.position_order, cfg.fa_type_order,
                    f"True Position vs true FA-type – {subset_name} (counts)",
                    cfg.out_dir / f"crosstab_true_both_{subset_name}.png",
                )
                _plot_crosstab(
                    lbl, "true_position", "pred_fa_type",
                    cfg.position_order, cfg.fa_type_order,
                    f"True Position vs predicted FA-type – {subset_name} (counts)",
                    cfg.out_dir / f"crosstab_truepos_predfa_{subset_name}.png",
                )
                _plot_crosstab(
                    lbl, "pred_position", "true_fa_type",
                    cfg.position_order, cfg.fa_type_order,
                    f"Predicted Position vs true FA-type – {subset_name} (counts)",
                    cfg.out_dir / f"crosstab_predpos_truefa_{subset_name}.png",
                )
                # Row-normalised versions
                _plot_crosstab(
                    lbl, "true_position", "true_fa_type",
                    cfg.position_order, cfg.fa_type_order,
                    f"True Position vs true FA-type – {subset_name} (row-normalised)",
                    cfg.out_dir / f"crosstab_true_both_norm_row_{subset_name}.png",
                    normalize="index",
                )

    # ------------------------------------------------------------------
    # 5. Save merged CSV
    # ------------------------------------------------------------------
    log.info("Step 5: Saving merged results CSV …")
    out_cols = [c for c in [
        "filename", "condition_name", "group", "split",
        "true_fa_type", "pred_fa_type", "max_prob_fa_type",
        "true_position", "pred_position", "max_prob_position",
        "umap1", "umap2",
    ] if c in df.columns]
    df[out_cols].to_csv(str(cfg.out_dir / "cross_classification_results.csv"), index=False)
    log.info("  Saved: cross_classification_results.csv  (%d rows)", len(df))
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Cross-classification visualization (UMAP + crosstabs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to the YAML configuration file.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print resolved config and exit without running.")
    p.add_argument("--log_level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--root_folder", default=None,
        help="Override root_folder for all paths. Useful when running on a different computer.",
    )
    args = p.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as fh:
        _raw = yaml.safe_load(fh)
    yaml_level = _raw.get("misc", {}).get("log_level", "INFO")
    _setup_logging(args.log_level or yaml_level)

    log.info("Loading config from: %s", args.config)
    cfg = load_config(args.config, root_folder=args.root_folder)

    if args.dry_run:
        print("\n=== DRY RUN – resolved CrossVisConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nRemove --dry_run to run for real.")
        return

    run_cross_vis(cfg)

    # Copy config to the output directory for reproducibility
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
