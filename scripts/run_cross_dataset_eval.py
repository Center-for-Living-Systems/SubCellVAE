#!/usr/bin/env python3
"""
Evaluate trained AE models on multiple datasets and generate
violin plots of MSE / L1 / Hessian-L1 reconstruction quality.

Datasets evaluated:
  vinc   — training dataset (reuses existing recon TIFs, no re-inference)
  pfak   — new dataset (runs model inference)
  ppax   — new dataset
  nih3t3 — new dataset

For vinc train/val patches, groups are split by FA type (annotation_label_name).
For external datasets, groups are dataset_condition.

One figure per variant is saved to <run_dir>/<variant>_cross_dataset.png,
with 3 subplots (one per metric).
A combined CSV is written to <run_dir>/cross_dataset_recon_metrics.csv.

Usage:
  python scripts/run_cross_dataset_eval.py <run_dir>
  python scripts/run_cross_dataset_eval.py <run_dir> \\
      --root-folder /home/lding/lding/fa_data_analysis \\
      --batch-size 256
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
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.modelling.dataset import PatchDataset


# ── constants ─────────────────────────────────────────────────────────────────

VARIANTS = [
    "semicon_both","baseline", "semisup_fa", "semisup_pos", "semisup_both",
    "conae", "semicon_fa", "semicon_pos", 
]


# (dataset_name, condition_name, relative patch path under root_folder)
EXTERNAL_DATASETS = [
    ("pfak",   "control", "ae_results/patches/cio_rb/pfak/control/tiff_patches32"),
    ("pfak",   "ycomp",   "ae_results/patches/cio_rb/pfak/ycomp/tiff_patches32"),
    ("ppax",   "control", "ae_results/patches/cio_rb/ppax/control/tiff_patches32"),
    ("ppax",   "ycomp",   "ae_results/patches/cio_rb/ppax/ycomp/tiff_patches32"),
    ("nih3t3", "control", "ae_results/patches/cio_rb/nih3t3/control/tiff_patches32"),
    ("nih3t3", "ycomp",   "ae_results/patches/cio_rb/nih3t3/ycomp/tiff_patches32"),
]

METRICS = ["recon_mse", "recon_l1", "recon_hessian_l1"]
METRIC_LABELS = {
    "recon_mse":        "MSE",
    "recon_l1":         "L1 (MAE)",
    "recon_hessian_l1": "Hessian L1",
}

# External dataset groups (fixed order at right side of x-axis)
EXTERNAL_GROUP_ORDER = [
    "pfak_control",  "pfak_ycomp",
    "ppax_control",  "ppax_ycomp",
    "nih3t3_control","nih3t3_ycomp",
]


# ── patch metrics ─────────────────────────────────────────────────────────────

def _patch_hessian_l1(raw: np.ndarray, recon: np.ndarray) -> float:
    if raw.ndim == 3:
        return float(np.mean([_patch_hessian_l1(raw[c], recon[c])
                               for c in range(raw.shape[0])]))
    r, p = raw.astype(np.float64), recon.astype(np.float64)
    Ixx_r = r[1:-1, 2:]  + r[1:-1, :-2] - 2 * r[1:-1, 1:-1]
    Ixx_p = p[1:-1, 2:]  + p[1:-1, :-2] - 2 * p[1:-1, 1:-1]
    Iyy_r = r[2:,  1:-1] + r[:-2, 1:-1] - 2 * r[1:-1, 1:-1]
    Iyy_p = p[2:,  1:-1] + p[:-2, 1:-1] - 2 * p[1:-1, 1:-1]
    Ixy_r = (r[2:, 2:] - r[2:, :-2] - r[:-2, 2:] + r[:-2, :-2]) / 4
    Ixy_p = (p[2:, 2:] - p[2:, :-2] - p[:-2, 2:] + p[:-2, :-2]) / 4
    H_raw   = np.sqrt(Ixx_r ** 2 + 2 * Ixy_r ** 2 + Iyy_r ** 2)
    H_recon = np.sqrt(Ixx_p ** 2 + 2 * Ixy_p ** 2 + Iyy_p ** 2)
    return float(np.mean(np.abs(H_raw - H_recon)))


def _metrics_from_arrays(raws: list[np.ndarray],
                          recons: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for r, p in zip(raws, recons):
        diff = r.astype(np.float64) - p.astype(np.float64)
        rows.append({
            "recon_mse":        float(np.mean(diff ** 2)),
            "recon_l1":         float(np.mean(np.abs(diff))),
            "recon_hessian_l1": _patch_hessian_l1(r, p),
        })
    return pd.DataFrame(rows)


# ── vinc: read from existing recon TIFs ───────────────────────────────────────

def _vinc_metrics(variant_dir: Path) -> pd.DataFrame | None:
    """Return patch-level metrics for vinc, with FA type groups for train/val."""
    recon_dir = variant_dir / "recon"
    raw_tif   = recon_dir / "patches_raw.tif"
    rec_tif   = recon_dir / "patches_recon.tif"
    idx_csv   = recon_dir / "patches_index.csv"
    if not (raw_tif.exists() and rec_tif.exists() and idx_csv.exists()):
        return None

    raw_all = tifffile.imread(str(raw_tif))
    rec_all = tifffile.imread(str(rec_tif))
    idx_df  = pd.read_csv(idx_csv)
    met_df  = _metrics_from_arrays(list(raw_all), list(rec_all))
    df = pd.concat([idx_df.reset_index(drop=True), met_df.reset_index(drop=True)],
                   axis=1)
    df["dataset"] = "vinc"

    # Try to merge FA type from latents.csv
    lat_csv = variant_dir / "latents.csv"
    fa_merged = False
    if lat_csv.exists():
        lat_df = pd.read_csv(lat_csv)
        if "annotation_label_name" in lat_df.columns:
            lat_df["_stem"] = lat_df["filename"].apply(lambda p: Path(p).stem)
            if "name" in df.columns:
                df["_stem"] = df["name"].apply(lambda p: Path(p).stem)
            else:
                df["_stem"] = df.index.astype(str)
            ann_map = (lat_df[["_stem", "annotation_label_name"]]
                       .dropna()
                       .drop_duplicates("_stem")
                       .set_index("_stem")["annotation_label_name"])
            df["fa_type"] = df["_stem"].map(ann_map)
            # only consider rows where annotation is a real label (not NaN / -1)
            has_fa = df["fa_type"].notna() & (df["fa_type"] != "-1")
            if has_fa.sum() > 0:
                fa_merged = True

    # Build group column
    if fa_merged:
        # vinc rows with FA label: group = "vinc_{split}_{fa_type}"
        df["group"] = df.apply(
            lambda r: (f"vinc_{r['split']}_{r['fa_type']}"
                       if (pd.notna(r.get("fa_type")) and r.get("fa_type") != "-1")
                       else f"vinc_{r['split']}_unlabeled"),
            axis=1,
        )
    else:
        df["group"] = df["dataset"] + "_" + df["condition_name"] + "_" + df["split"]

    return df


# ── external dataset: run inference ──────────────────────────────────────────

def _infer_dataset(model, patch_dir: Path, device: str,
                   batch_size: int) -> tuple[list, list]:
    """Run model on all patches in patch_dir; return (raws, recons)."""
    ds = PatchDataset(str(patch_dir), condition=0, condition_name="")
    if len(ds) == 0:
        return [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0)

    cls_name = type(model).__name__
    if "SemiSup" in cls_name:
        model_type = "semisup"
    elif "Contrastive" in cls_name:
        model_type = "contrastive"
    elif "VAE" in cls_name:
        model_type = "vae"
    else:
        model_type = "ae"

    raws, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            if x.dim() == 3:          # (B, H, W) → (B, 1, H, W)
                x = x.unsqueeze(1)
            if model_type == "vae":
                x_hat, mu, _, _ = model(x)
            elif model_type == "semisup":
                x_hat, _, _ = model(x)
            else:
                x_hat, _ = model(x)
            for raw_p, rec_p in zip(x.cpu().numpy(), x_hat.cpu().numpy()):
                if raw_p.shape[0] == 1:
                    raw_p, rec_p = raw_p[0], rec_p[0]
                raws.append(raw_p.astype(np.float32))
                recons.append(rec_p.astype(np.float32))
    return raws, recons


def _external_metrics(model, dataset_name: str, condition_name: str,
                      patch_dir: Path, device: str,
                      batch_size: int) -> pd.DataFrame | None:
    if not patch_dir.exists():
        print(f"    [skip] patch dir not found: {patch_dir}")
        return None
    print(f"    inference on {dataset_name}/{condition_name} "
          f"({len(list(patch_dir.glob('*.tif')))} patches) …", flush=True)
    raws, recons = _infer_dataset(model, patch_dir, device, batch_size)
    if not raws:
        return None
    met_df = _metrics_from_arrays(raws, recons)
    met_df["dataset"]        = dataset_name
    met_df["condition_name"] = condition_name
    met_df["split"]          = "test"
    met_df["group"]          = dataset_name + "_" + condition_name
    return met_df


# ── plotting ──────────────────────────────────────────────────────────────────

_LABEL_SUBS = [
    ("nascent adhesion",   "Nas Adh"),
    ("no adhesion",        "No Adh"),
    ("focal complex",      "Foc Cpx"),
    ("focal adhesion",     "Foc Adh"),
    ("fibrillar adhesion", "Fib Adh"),
    ("stress fiber",       "Str Fib"),
    ("unlabeled",          "unlbl"),
    ("nih3t3",             "ds4"),
    ("pfak",               "ds2"),
    ("ppax",               "ds3"),
    ("vinc",               "ds1"),
    ("control",            "ctrl"),
    ("ycomp",              "yc"),
    ("train",              "tr"),
    ("val",                "val"),
    ("test",               "tst"),
]

def _shorten(label: str) -> str:
    """Abbreviate long group labels for compact x-axis display."""
    import re
    result = label
    for long, short in _LABEL_SUBS:
        result = re.sub(re.escape(long), short, result, flags=re.IGNORECASE)
    return result


def _build_group_order(variant_df: pd.DataFrame) -> list[str]:
    """Return ordered x-axis group list: vinc FA groups first, then external."""
    # vinc groups: sorted by split then fa_type
    vinc_groups = sorted(
        g for g in variant_df["group"].unique()
        if g.startswith("vinc_")
    )
    ext_groups = [g for g in EXTERNAL_GROUP_ORDER
                  if g in variant_df["group"].values]
    return vinc_groups + ext_groups


def _violin_plot_single(variant_df: pd.DataFrame, variant_name: str,
                         metric: str, save_path: Path) -> None:
    """One figure per variant per metric, x = groups."""
    group_order = _build_group_order(variant_df)
    if not group_order:
        return

    sub = variant_df[variant_df[metric].notna()].copy()
    present = [g for g in group_order if g in sub["group"].values]
    if sub.empty or not present:
        return

    display_labels = [_shorten(g) for g in present]

    fig, ax = plt.subplots(figsize=(max(12, len(present) * 1.6), 9.5))
    sns.violinplot(data=sub, x="group", y=metric,
                   order=present, ax=ax,
                   inner="box", cut=0, linewidth=0.8, width=1.2,
                   color="cornflowerblue")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=40)
    ax.tick_params(axis="y", labelsize=36)
    # ax.set_title(f"{variant_name} — {METRIC_LABELS[metric]}", fontsize=36)
    ax.set_xlabel("")
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=36)

    # clip y-axis at 99th percentile to avoid outlier whitespace
    y99 = float(sub[metric].quantile(0.99))
    ymin = float(sub[metric].quantile(0.001))
    ax.set_ylim(max(0, ymin * 0.95), y99 * 1.05)

    # vertical separator between vinc and external groups
    n_vinc = sum(1 for g in present if g.startswith("vinc_"))
    if 0 < n_vinc < len(present):
        ax.axvline(n_vinc - 0.5, color="grey", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"  saved → {save_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root-folder",
                        default="/home/lding/lding/fa_data_analysis")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir     = args.run_dir
    root_folder = Path(args.root_folder)
    batch_size  = args.batch_size
    device      = ("cuda" if torch.cuda.is_available() else "cpu") \
                  if args.device == "auto" else args.device

    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    print(f"Run dir    : {run_dir}")
    print(f"Device     : {device}")
    print(f"Batch size : {batch_size}")
    print()

    all_rows = []

    for variant in VARIANTS:
        variant_dir = run_dir / variant
        model_pt    = variant_dir / "model_final.pt"
        if not variant_dir.is_dir() or not model_pt.exists():
            print(f"[skip] {variant} — model_final.pt not found")
            continue

        print(f"── {variant} ──────────────────────────────")
        variant_rows = []

        # 1. vinc from existing recon TIFs
        vinc_df = _vinc_metrics(variant_dir)
        if vinc_df is not None:
            vinc_df["variant"] = variant
            variant_rows.append(vinc_df)
            fa_groups = vinc_df["group"].nunique()
            print(f"  vinc: {len(vinc_df)} patches, {fa_groups} groups")
        else:
            print(f"  vinc: recon TIFs not found — skipping")

        # 2. external datasets — load model once per variant
        print(f"  loading model …", flush=True)
        model = torch.load(str(model_pt), map_location=device, weights_only=False)
        model.eval()

        for ds_name, cond_name, rel_path in EXTERNAL_DATASETS:
            patch_dir = root_folder / rel_path
            ext_df = _external_metrics(model, ds_name, cond_name,
                                        patch_dir, device, batch_size)
            if ext_df is not None:
                ext_df["variant"] = variant
                variant_rows.append(ext_df)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

        if variant_rows:
            variant_df = pd.concat(variant_rows, ignore_index=True)
            all_rows.append(variant_df)

            # one figure per metric
            for metric in METRICS:
                save_path = run_dir / f"{variant}_cross_dataset_{metric}.png"
                _violin_plot_single(variant_df, variant, metric, save_path)
        print()

    if not all_rows:
        sys.exit("No results collected.")

    combined = pd.concat(all_rows, ignore_index=True)
    out_csv = run_dir / "cross_dataset_recon_metrics.csv"
    combined.to_csv(out_csv, index=False)
    print(f"Combined metrics → {out_csv}  ({len(combined)} rows)")
    print("\nDone.")


if __name__ == "__main__":
    main()
