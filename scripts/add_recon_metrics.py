#!/usr/bin/env python3
"""
Compute recon_l1 and recon_hessian_l1 for already-trained models and
generate distribution / group box plots — no model reload needed.

Reads:
  <variant>/recon/patches_raw.tif
  <variant>/recon/patches_recon.tif
  <variant>/recon/patches_index.csv
  <variant>/latents.csv   (for condition / annotation metadata)

Writes per variant:
  recon_l1_distribution.png
  recon_l1_by_condition_split.png
  recon_l1_by_annotation_split.png        (if annotation labels present)
  hessian_l1_distribution.png
  hessian_l1_by_condition_split.png
  hessian_l1_by_annotation_split.png      (if annotation labels present)
  recon_metrics.csv                        (patch-level l1 + hessian_l1)

Usage:
  python scripts/add_recon_metrics.py <run_dir>
  python scripts/add_recon_metrics.py <run_dir> --boxplot-kind violin
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile


# ── Hessian helper ────────────────────────────────────────────────────────────

def _patch_hessian_l1(raw: np.ndarray, recon: np.ndarray) -> float:
    """Mean absolute difference of Hessian maps between raw and recon patches.

    For each interior pixel computes the Frobenius norm of the 2×2 Hessian
    matrix as a scalar map, then returns mean(|H_raw - H_recon|).
    For multi-channel (C,H,W) averages over channels.
    """
    if raw.ndim == 3:
        return float(np.mean([_patch_hessian_l1(raw[c], recon[c])
                               for c in range(raw.shape[0])]))
    r = raw.astype(np.float64)
    p = recon.astype(np.float64)
    Ixx_r = r[1:-1, 2:]  + r[1:-1, :-2] - 2 * r[1:-1, 1:-1]
    Ixx_p = p[1:-1, 2:]  + p[1:-1, :-2] - 2 * p[1:-1, 1:-1]
    Iyy_r = r[2:,  1:-1] + r[:-2, 1:-1] - 2 * r[1:-1, 1:-1]
    Iyy_p = p[2:,  1:-1] + p[:-2, 1:-1] - 2 * p[1:-1, 1:-1]
    Ixy_r = (r[2:, 2:] - r[2:, :-2] - r[:-2, 2:] + r[:-2, :-2]) / 4
    Ixy_p = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / 4
    # scalar Hessian map = Frobenius norm of [[Ixx,Ixy],[Ixy,Iyy]] per pixel
    H_raw   = np.sqrt(Ixx_r ** 2 + 2 * Ixy_r ** 2 + Iyy_r ** 2)
    H_recon = np.sqrt(Ixx_p ** 2 + 2 * Ixy_p ** 2 + Iyy_p ** 2)
    # per-pixel absolute difference, then mean
    return float(np.mean(np.abs(H_raw - H_recon)))


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _distribution_plot(values: np.ndarray, title: str, xlabel: str,
                       save_path: Path) -> None:
    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(values, bins=60, edgecolor="none", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _group_plot(df: pd.DataFrame, metric_col: str, group_col: str,
                group_order: list, title: str, save_path: Path,
                kind: str = "box") -> None:
    sub = df[df[metric_col].notna() & np.isfinite(df[metric_col])].copy()
    hue_order = [s for s in ["train", "val"] if s in sub["split"].values]
    use_hue = len(hue_order) > 0
    n_groups = len(group_order)
    fig, ax = plt.subplots(figsize=(max(5, n_groups * 2.5), 4))
    if kind == "violin":
        sns.violinplot(data=sub, x=group_col, y=metric_col,
                       hue="split" if use_hue else None,
                       hue_order=hue_order if use_hue else None,
                       order=group_order, ax=ax, inner="box", split=False)
    else:
        sns.boxplot(data=sub, x=group_col, y=metric_col,
                    hue="split" if use_hue else None,
                    hue_order=hue_order if use_hue else None,
                    order=group_order, ax=ax, legend=False)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_order, rotation=40, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric_col)
    if use_hue:
        ax.legend(title="split", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


# ── Per-variant logic ─────────────────────────────────────────────────────────

def process_variant(variant_dir: Path, kind: str) -> None:
    recon_dir = variant_dir / "recon"
    raw_tif   = recon_dir / "patches_raw.tif"
    rec_tif   = recon_dir / "patches_recon.tif"
    idx_csv   = recon_dir / "patches_index.csv"
    lat_csv   = variant_dir / "latents.csv"

    if not raw_tif.exists() or not rec_tif.exists():
        print(f"  [skip] no patches_raw/recon.tif in {variant_dir.name}")
        return
    if not idx_csv.exists():
        print(f"  [skip] no patches_index.csv in {variant_dir.name}")
        return

    print(f"  {variant_dir.name}: loading TIFs …", flush=True)
    raw_all = tifffile.imread(str(raw_tif))   # (N, [C,] H, W)
    rec_all = tifffile.imread(str(rec_tif))
    idx_df  = pd.read_csv(idx_csv)            # frame, split, condition, condition_name, group, name

    N = raw_all.shape[0]
    print(f"  {variant_dir.name}: computing metrics for {N} patches …", flush=True)

    l1_vals  = []
    hess_vals = []
    for i in range(N):
        r = raw_all[i].astype(np.float32)
        p = rec_all[i].astype(np.float32)
        diff = r - p
        l1_vals.append(float(np.mean(np.abs(diff))))
        hess_vals.append(_patch_hessian_l1(r, p))

    idx_df["recon_l1"]         = l1_vals
    idx_df["recon_hessian_l1"] = hess_vals

    # save patch-level metrics
    out_metrics = variant_dir / "recon_metrics.csv"
    idx_df.to_csv(out_metrics, index=False)

    # merge annotation labels + mse from latents.csv if available
    lat_df = None
    if lat_csv.exists():
        lat_df = pd.read_csv(lat_csv)
        lat_df["_name"] = lat_df["filename"].apply(lambda p: Path(p).stem)
        merge_cols = ["_name"]
        for c in lat_df.columns:
            if c.startswith("annotation") or c in ("recon_mse", "norm_mse"):
                merge_cols.append(c)
        merge_df = lat_df[merge_cols].drop_duplicates("_name")
        idx_df = idx_df.merge(merge_df, left_on="name", right_on="_name", how="left")

    # ── build ordered lists for plots ────────────────────────────────────
    cond_order = sorted(idx_df["condition_name"].dropna().unique().tolist())

    # annotation labels (fa type)
    has_ann = ("annotation_label" in idx_df.columns and
               (idx_df["annotation_label"].fillna(-1) != -1).any())
    if has_ann:
        ann_order = (idx_df[idx_df["annotation_label"].fillna(-1) != -1]
                     ["annotation_label_name"].dropna().unique().tolist())
    has_ann2 = ("annotation_label_2" in idx_df.columns and
                (idx_df["annotation_label_2"].fillna(-1) != -1).any())
    if has_ann2:
        ann2_order = (idx_df[idx_df["annotation_label_2"].fillna(-1) != -1]
                      ["annotation_label_2_name"].dropna().unique().tolist())

    # ── MSE plots (from latents.csv) ─────────────────────────────────────
    for metric, label in [("recon_mse",  "Reconstruction MSE"),
                           ("norm_mse",   "Normalised MSE  (MSE / mean intensity)")]:
        if metric not in idx_df.columns:
            continue
        _distribution_plot(
            idx_df[metric].dropna().values, f"{label} — all patches",
            label, variant_dir / f"{metric}_distribution.png")
        _group_plot(idx_df, metric, "condition_name", cond_order,
                    f"{label} by condition × split",
                    variant_dir / f"{metric}_by_condition_split.png", kind=kind)
        if has_ann:
            df_ann = idx_df[idx_df["annotation_label"].fillna(-1) != -1]
            _group_plot(df_ann, metric, "annotation_label_name", ann_order,
                        f"{label} by FA type × split",
                        variant_dir / f"{metric}_by_annotation_split.png", kind=kind)
        if has_ann2:
            df_ann2 = idx_df[idx_df["annotation_label_2"].fillna(-1) != -1]
            _group_plot(df_ann2, metric, "annotation_label_2_name", ann2_order,
                        f"{label} by position × split",
                        variant_dir / f"{metric}_by_annotation2_split.png", kind=kind)

    # ── L1 plots ─────────────────────────────────────────────────────────
    _distribution_plot(
        np.array(l1_vals), "Reconstruction L1 (MAE) — all patches",
        "Reconstruction L1", variant_dir / "recon_l1_distribution.png")
    _group_plot(idx_df, "recon_l1", "condition_name", cond_order,
                "Reconstruction L1 by condition × split",
                variant_dir / "recon_l1_by_condition_split.png", kind=kind)
    if has_ann:
        df_ann = idx_df[idx_df["annotation_label"].fillna(-1) != -1]
        _group_plot(df_ann, "recon_l1", "annotation_label_name", ann_order,
                    "Reconstruction L1 by FA type × split",
                    variant_dir / "recon_l1_by_annotation_split.png", kind=kind)
    if has_ann2:
        df_ann2 = idx_df[idx_df["annotation_label_2"].fillna(-1) != -1]
        _group_plot(df_ann2, "recon_l1", "annotation_label_2_name", ann2_order,
                    "Reconstruction L1 by position × split",
                    variant_dir / "recon_l1_by_annotation2_split.png", kind=kind)

    # ── Hessian L1 plots ─────────────────────────────────────────────────
    _distribution_plot(
        np.array(hess_vals), "Hessian L1 — all patches",
        "Hessian L1  (|ΔIxx|+|ΔIyy|+2|ΔIxy|, interior pixels)",
        variant_dir / "hessian_l1_distribution.png")
    _group_plot(idx_df, "recon_hessian_l1", "condition_name", cond_order,
                "Hessian L1 by condition × split",
                variant_dir / "hessian_l1_by_condition_split.png", kind=kind)
    if has_ann:
        df_ann = idx_df[idx_df["annotation_label"].fillna(-1) != -1]
        _group_plot(df_ann, "recon_hessian_l1", "annotation_label_name", ann_order,
                    "Hessian L1 by FA type × split",
                    variant_dir / "hessian_l1_by_annotation_split.png", kind=kind)
    if has_ann2:
        df_ann2 = idx_df[idx_df["annotation_label_2"].fillna(-1) != -1]
        _group_plot(df_ann2, "recon_hessian_l1", "annotation_label_2_name", ann2_order,
                    "Hessian L1 by position × split",
                    variant_dir / "hessian_l1_by_annotation2_split.png", kind=kind)

    print(f"  {variant_dir.name}: done  "
          f"l1_mean={np.mean(l1_vals):.4f}  hessian_l1_mean={np.mean(hess_vals):.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--boxplot-kind", default="box", choices=["box", "violin"])
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    print(f"Run dir: {run_dir}")
    for variant_dir in sorted(run_dir.iterdir()):
        if variant_dir.is_dir():
            process_variant(variant_dir, args.boxplot_kind)


if __name__ == "__main__":
    main()
