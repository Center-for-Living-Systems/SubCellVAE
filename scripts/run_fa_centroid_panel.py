#!/usr/bin/env python3
"""
Panel of N patches nearest to the latent centroid of each FA type,
drawn from the semisup_both (or any) variant of a trained run.

Rows : FA types (No adhesion excluded)
Cols : N patches closest to the FA-type centroid in latent space

Patch images are loaded from <variant_dir>/recon/patches/raw_{split}_{filename}.

Usage:
  python scripts/run_fa_centroid_panel.py <run_dir>
  python scripts/run_fa_centroid_panel.py <run_dir> --variant semisup_both --n 6
  python scripts/run_fa_centroid_panel.py <run_dir> --split train
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

sys.path.insert(0, str(Path(__file__).parents[1]))


EXCLUDE_FA = {"no adhesion"}

FA_SHORT = {
    "nascent adhesion":   "Nas Adh",
    "focal complex":      "Foc Cpx",
    "focal adhesion":     "Foc Adh",
    "fibrillar adhesion": "Fib Adh",
    "stress fiber":       "Str Fib",
}


def _shorten_fa(name: str) -> str:
    return FA_SHORT.get(name.lower(), name)


def _load_patch(patch_dir: Path, split: str, filename: str) -> np.ndarray | None:
    """Load raw patch TIF; returns 2-D (H,W) float32 array or None."""
    p = patch_dir / f"raw_{split}_{filename}"
    if not p.exists():
        return None
    img = tifffile.imread(str(p)).astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--variant", default="semisup_both")
    parser.add_argument("--n", type=int, default=6,
                        help="Patches per FA type (columns)")
    parser.add_argument("--pool", type=int, default=40,
                        help="Candidate pool: randomly sample --n from the "
                             "--pool patches nearest the centroid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "all"])
    args = parser.parse_args()

    variant_dir = args.run_dir / args.variant
    lat_csv     = variant_dir / "latents.csv"
    patch_dir   = variant_dir / "recon" / "patches"

    if not lat_csv.exists():
        sys.exit(f"latents.csv not found: {lat_csv}")
    if not patch_dir.is_dir():
        sys.exit(f"patch dir not found: {patch_dir}")

    df = pd.read_csv(lat_csv)

    # latent columns (z_0 … z_N or lat_0 … lat_N)
    lat_cols = [c for c in df.columns if c.startswith("z_") or c.startswith("lat_")]
    if not lat_cols:
        sys.exit("No latent columns (z_* / lat_*) found in latents.csv")
    print(f"Latent dims : {len(lat_cols)}")

    # filter split
    if args.split != "all" and "split" in df.columns:
        df = df[df["split"] == args.split]
    print(f"Patches ({args.split}): {len(df)}")

    # keep annotated rows, exclude No Adhesion
    df = df[df["annotation_label_name"].notna()]
    df = df[df["annotation_label"].fillna(-1).astype(float) != -1]
    df = df[~df["annotation_label_name"].str.lower().isin(EXCLUDE_FA)]

    fa_types = sorted(df["annotation_label_name"].unique(),
                      key=lambda s: s.lower())
    print(f"FA types    : {fa_types}")

    if not fa_types:
        sys.exit("No annotated FA patches found after filtering.")

    n_rows = len(fa_types)
    n_cols = args.n

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.4),
        gridspec_kw={"hspace": 0.05, "wspace": 0.03},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    rng = np.random.default_rng(args.seed)

    for row_idx, fa_type in enumerate(fa_types):
        sub = df[df["annotation_label_name"] == fa_type].copy()
        lat_mat  = sub[lat_cols].values.astype(np.float32)
        centroid = lat_mat.mean(axis=0)
        dists    = np.linalg.norm(lat_mat - centroid, axis=1)

        # pool = closest --pool patches; sample --n randomly from that neighborhood
        pool_size = min(args.pool, len(sub))
        pool_idx  = np.argsort(dists)[:pool_size]
        chosen    = rng.choice(pool_idx, size=min(n_cols, pool_size), replace=False)
        nearest   = sub.iloc[chosen]

        label = _shorten_fa(fa_type)
        for col_idx, (_, row) in enumerate(nearest.iterrows()):
            ax = axes[row_idx, col_idx]
            split_val = row.get("split", args.split)
            img = _load_patch(patch_dir, split_val, row["filename"])
            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.axis("off")
            else:
                ax.imshow(img, cmap="gray", interpolation="nearest",
                          vmin=np.percentile(img, 1),
                          vmax=np.percentile(img, 99))
                ax.axis("off")

            if col_idx == 0:
                ax.set_ylabel(label, fontsize=13, rotation=90,
                              labelpad=6, va="center")
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=False)

        # column headers (rank) on top row
        if row_idx == 0:
            for col_idx in range(n_cols):
                axes[0, col_idx].set_title(f"#{col_idx + 1}", fontsize=10)

    run_name = args.run_dir.name
    fig.suptitle(
        f"{run_name} / {args.variant}  —  {args.split} patches nearest FA centroid",
        fontsize=13, y=1.01,
    )

    save_path = args.run_dir / f"{args.variant}_fa_centroid_panel.png"
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
