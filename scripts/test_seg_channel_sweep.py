#!/usr/bin/env python3
"""
Sweep segmentation channels (0-3) on vinc control images.
For each channel, saves one overlay PNG per image into a separate output folder.

Overlay: segmentation channel (binned, grey) with cell mask in red.
Stats summary printed at the end.

Output:
  ae_results/test_seg_channel_sweep/ch{N}/  (N = 0..3)
"""

import sys
from pathlib import Path

import czifile
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import downscale_local_mean

ROOT    = Path("/home/lding/lding/fa_data_analysis")
IMG_DIR = ROOT / ("fa_data/other_paxillin/"
                  "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/"
                  "Control")
OUT_BASE = ROOT / "ae_results/test_seg_channel_sweep"

TARGET      = 512
SEG_CHS     = [0, 1, 2, 3]
THRESHOLD   = 0.1
CLOSE_SIZE  = 11
MIN_FINAL   = 30_000

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.dataprep.patch_prep import segment_cell_mask


def load_all_channels_binned(path: Path, target: int = 512):
    raw = czifile.imread(str(path)).squeeze()    # (C, H, W) uint16
    H, W = raw.shape[1], raw.shape[2]
    fh = max(1, H // target)
    fw = max(1, W // target)
    binned = []
    for ch in range(raw.shape[0]):
        binned.append(downscale_local_mean(raw[ch].astype(float), (fh, fw)))
    return binned, (fh, fw)   # list of (512,512) float arrays, still uint16-range


def segment_channel(ch_img: np.ndarray, bin_factor) -> np.ndarray:
    fh, fw = bin_factor
    min_final = max(100, MIN_FINAL // (fh * fw))
    return segment_cell_mask(
        ch_img,
        threshold=THRESHOLD,
        close_size=CLOSE_SIZE,
        min_size_initial=3,
        min_size_post_close=10,
        min_size_final=min_final,
    )


def percentile_stretch(img: np.ndarray, lo=1, hi=99) -> np.ndarray:
    p1, p99 = np.percentile(img, lo), np.percentile(img, hi)
    if p99 <= p1:
        return np.zeros_like(img)
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def save_overlay(ch_img: np.ndarray, mask: np.ndarray, out_path: Path, title: str):
    display = percentile_stretch(ch_img)
    rgb = np.stack([display, display, display], axis=-1)
    # red overlay for cell mask
    rgb[mask > 0, 0] = np.clip(rgb[mask > 0, 0] * 0.5 + 0.5, 0, 1)
    rgb[mask > 0, 1] = rgb[mask > 0, 1] * 0.5
    rgb[mask > 0, 2] = rgb[mask > 0, 2] * 0.5

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb, interpolation='nearest')
    inside_pct = 100 * (mask > 0).sum() / mask.size
    ax.set_title(f'{title}\ninside={inside_pct:.1f}%', fontsize=8)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    czis = sorted(IMG_DIR.glob("*.czi"))
    print(f"Found {len(czis)} CZI files")

    # create output dirs
    out_dirs = {}
    for ch in SEG_CHS:
        d = OUT_BASE / f"ch{ch}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[ch] = d

    # stats: ch → list of inside fractions
    stats = {ch: [] for ch in SEG_CHS}

    for idx, czi_path in enumerate(czis):
        stem = czi_path.stem
        short = f"{idx+1:02d}"
        print(f"[{idx+1:2d}/{len(czis)}] {stem[-30:]}", flush=True)

        try:
            channels, bin_factor = load_all_channels_binned(czi_path, TARGET)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for ch in SEG_CHS:
            if ch >= len(channels):
                print(f"  ch{ch}: not available (only {len(channels)} channels)")
                continue

            ch_img = channels[ch]
            mask   = segment_channel(ch_img, bin_factor)
            inside_pct = 100 * (mask > 0).sum() / mask.size
            stats[ch].append(inside_pct)

            title = f"img {short}  seg_ch={ch}"
            out_path = out_dirs[ch] / f"{short}_{stem[-20:]}.png"
            save_overlay(ch_img, mask, out_path, title)
            print(f"  ch{ch}: inside={inside_pct:.1f}%", end="   ")
        print()

    # summary
    print("\n=== Summary: mean inside% per channel ===")
    for ch in SEG_CHS:
        vals = stats[ch]
        if vals:
            print(f"  ch{ch}: mean={np.mean(vals):.1f}%  "
                  f"min={np.min(vals):.1f}%  max={np.max(vals):.1f}%  "
                  f"n={len(vals)}")

    print(f"\nOutputs in: {OUT_BASE}")
    for ch in SEG_CHS:
        print(f"  ch{ch}/  →  {out_dirs[ch]}")


if __name__ == '__main__':
    main()
