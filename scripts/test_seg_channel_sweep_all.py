#!/usr/bin/env python3
"""
Segmentation channel sweep (0-3) across all 4 datasets × 2 conditions.

For each dataset/condition, samples up to MAX_IMAGES CZI files and runs
segmentation with each channel. Saves one overlay PNG per image per channel.

Output:
  ae_results/test_seg_channel_sweep_all/{dataset}_{condition}/ch{N}/
  ae_results/test_seg_channel_sweep_all/summary.txt
"""

import sys
from pathlib import Path

import czifile
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import downscale_local_mean

ROOT      = Path("/home/lding/lding/fa_data_analysis")
DATA_BASE = ROOT / "fa_data/other_paxillin"
OUT_BASE  = ROOT / "ae_results/test_seg_channel_sweep_all"

DATASETS = {
    'vinc': {
        'subdir'  : '20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568',
        'cond_dir': {'control': 'Control', 'ycomp': 'Ycomp'},
    },
    'ppax': {
        'subdir'  : '20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568',
        'cond_dir': {'control': 'Control', 'ycomp': 'Y-comp'},
    },
    'pfak': {
        'subdir'  : '20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025',
        'cond_dir': {'control': 'Control', 'ycomp': 'Ycomp'},
    },
    'nih3t3': {
        'subdir'  : '20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH',
        'cond_dir': {'control': 'Control', 'ycomp': 'YCompound'},
    },
}

SEG_CHS    = [0, 1, 2, 3]
TARGET     = 512
MAX_IMAGES = 9999    # all images
THRESHOLD  = 0.1
CLOSE_SIZE = 11
MIN_FINAL  = 30_000

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.dataprep.patch_prep import segment_cell_mask


def load_channels_binned(path: Path, target: int = 512):
    raw = czifile.imread(str(path)).squeeze()
    if raw.ndim == 2:
        raise ValueError(f"2-D image (single channel?): {path.name}")
    H, W = raw.shape[1], raw.shape[2]
    fh = max(1, H // target)
    fw = max(1, W // target)
    return [downscale_local_mean(raw[c].astype(float), (fh, fw))
            for c in range(raw.shape[0])], (fh, fw)


def segment(ch_img: np.ndarray, bin_factor) -> np.ndarray:
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
        return np.zeros_like(img, dtype=float)
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def save_overlay(ch_img, mask, out_path, title):
    disp = percentile_stretch(ch_img)
    rgb  = np.stack([disp, disp, disp], axis=-1)
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
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_lines = []

    for ds_name, ds_info in DATASETS.items():
        for cond, cond_dir in ds_info['cond_dir'].items():
            img_dir = DATA_BASE / ds_info['subdir'] / cond_dir
            if not img_dir.exists():
                print(f"[SKIP] not found: {img_dir}")
                continue

            czis = sorted(img_dir.glob("*.czi"))[:MAX_IMAGES]
            if not czis:
                print(f"[SKIP] no CZI files in {img_dir}")
                continue

            tag = f"{ds_name}_{cond}"
            print(f"\n{'='*60}")
            print(f" {tag}  ({len(czis)} images)")
            print(f"{'='*60}")

            # create per-channel output dirs
            out_dirs = {}
            for ch in SEG_CHS:
                d = OUT_BASE / tag / f"ch{ch}"
                d.mkdir(parents=True, exist_ok=True)
                out_dirs[ch] = d

            ch_inside = {ch: [] for ch in SEG_CHS}

            for idx, czi_path in enumerate(czis):
                print(f"  [{idx+1}/{len(czis)}] {czi_path.name[-40:]}", flush=True)
                try:
                    channels, bin_factor = load_channels_binned(czi_path, TARGET)
                except Exception as e:
                    print(f"    SKIP: {e}")
                    continue

                n_ch = len(channels)
                for ch in SEG_CHS:
                    if ch >= n_ch:
                        print(f"    ch{ch}: N/A (only {n_ch} channels)")
                        continue
                    mask = segment(channels[ch], bin_factor)
                    pct  = 100 * (mask > 0).sum() / mask.size
                    ch_inside[ch].append(pct)

                    out_path = out_dirs[ch] / f"{idx+1:02d}_{czi_path.stem[-20:]}.png"
                    save_overlay(channels[ch], mask, out_path,
                                 f"{tag}  img{idx+1}  seg_ch={ch}")
                    print(f"    ch{ch}: {pct:.1f}%", end="   ")
                print()

            # summary for this dataset/condition
            summary_lines.append(f"\n{tag}")
            for ch in SEG_CHS:
                vals = ch_inside[ch]
                if vals:
                    line = (f"  ch{ch}: mean={np.mean(vals):.1f}%  "
                            f"min={np.min(vals):.1f}%  max={np.max(vals):.1f}%")
                else:
                    line = f"  ch{ch}: N/A"
                print(line)
                summary_lines.append(line)

    # write summary file
    summary_path = OUT_BASE / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written → {summary_path}")
    print(f"Overlays in     → {OUT_BASE}")


if __name__ == '__main__':
    main()
