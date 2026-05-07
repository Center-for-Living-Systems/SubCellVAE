#!/usr/bin/env python3
"""
Test CIO normalization on vinc control images.

Pipeline per image:
  1. Load CZI, extract ch=1 (pax)  +  ch=0 (seg)
  2. Bin both to 512×512 (block-average)
  3. Apply rolling ball r=5 to pax channel
  4. Segment cell from ch=0 (binned)
  5. CIO normalise: (img - mean_outside) / (mean_inside - mean_outside)
  6. Collect inside-cell pixel values

Output: overlaid histogram of normalised intensities, one line per image.
"""

import sys
from pathlib import Path

import czifile
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import rolling_ball
from skimage.transform import downscale_local_mean

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path("/home/lding/lding/fa_data_analysis")
IMG_DIR   = ROOT / ("fa_data/other_paxillin/"
                    "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/"
                    "Control")
OUT_DIR   = ROOT / "ae_results/test_cio_rb_histogram"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAX_CH  = 1
SEG_CH  = 0
TARGET  = 512          # bin to this size (both dims)
RB_R    = 5            # rolling ball radius on binned image

# ── segmentation (reuse parameters from patchprep configs) ────────────────────
SEG_THRESHOLD  = 0.1
SEG_CLOSE_SIZE = 11
SEG_MIN_FINAL  = 30_000   # in full-res pixels; scale down for binned

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.dataprep.patch_prep import segment_cell_mask


def load_and_bin(path: Path, target: int = 512):
    """Load CZI, return (pax_binned_float, seg_binned_uint16, bin_factor).

    pax channel: divided by 255² then binned (float [0,~1]).
    seg channel: kept as raw uint16 then binned — so segment_cell_mask can
                 apply its own percentile stretch correctly.
    """
    raw = czifile.imread(str(path)).squeeze()
    if raw.ndim == 2:
        raise ValueError(f"Expected multi-channel image, got 2-D: {path.name}")

    pax    = raw[PAX_CH].astype(float) / (255 * 255)
    seg_ch = raw[SEG_CH].astype(float)          # raw uint16-range for seg

    H, W = pax.shape
    fh = max(1, H // target)
    fw = max(1, W // target)
    pax_b    = downscale_local_mean(pax,    (fh, fw))
    seg_ch_b = downscale_local_mean(seg_ch, (fh, fw))   # still uint16-range
    return pax_b, seg_ch_b, (fh, fw)


def apply_rb(img: np.ndarray, radius: float) -> np.ndarray:
    bg = rolling_ball(img, radius=radius)
    return img - bg


def cio_normalise(img: np.ndarray, seg: np.ndarray):
    """(img - mean_outside) / (mean_inside - mean_outside), no clip, no scale."""
    outside = seg == 0
    inside  = seg > 0
    mean_out = float(np.mean(img[outside])) if outside.any() else 0.0
    mean_in  = float(np.mean(img[inside]))  if inside.any()  else mean_out + 1.0
    denom = mean_in - mean_out
    if denom == 0.0:
        denom = 1.0
    return (img - mean_out) / denom, mean_out, mean_in


def segment_binned(seg_ch_b: np.ndarray, bin_factor) -> np.ndarray:
    """Segment on the binned channel; adjust min_size for binned pixel area."""
    fh, fw = bin_factor
    min_final = max(100, SEG_MIN_FINAL // (fh * fw))
    seg = segment_cell_mask(
        seg_ch_b,
        threshold=SEG_THRESHOLD,
        close_size=SEG_CLOSE_SIZE,
        min_size_initial=3,
        min_size_post_close=10,
        min_size_final=min_final,
    )
    return seg


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    czis = sorted(IMG_DIR.glob("*.czi"))
    print(f"Found {len(czis)} CZI files in {IMG_DIR.name}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_all, ax_inside = axes
    cmap = plt.cm.tab20
    colours = [cmap(i / max(len(czis) - 1, 1)) for i in range(len(czis))]

    stats = []
    for idx, czi_path in enumerate(czis):
        print(f"  [{idx+1:2d}/{len(czis)}] {czi_path.name} …", end=" ", flush=True)
        try:
            pax_b, seg_ch_b, bin_factor = load_and_bin(czi_path, TARGET)
        except Exception as e:
            print(f"SKIP ({e})")
            continue

        pax_rb  = apply_rb(pax_b, RB_R)
        seg     = segment_binned(seg_ch_b, bin_factor)
        norm, mean_out, mean_in = cio_normalise(pax_rb, seg)

        all_vals    = norm.ravel()
        inside_vals = norm[seg > 0]

        col   = colours[idx]
        label = czi_path.stem[-8:]   # last 8 chars to keep label short

        # all-pixel histogram
        ax_all.hist(all_vals, bins=200, range=(-0.5, 3.0),
                    density=True, histtype='step', color=col, alpha=0.7, label=label)

        # inside-cell histogram
        if len(inside_vals):
            ax_inside.hist(inside_vals, bins=200, range=(-0.5, 3.0),
                           density=True, histtype='step', color=col, alpha=0.7, label=label)

        n_inside = int((seg > 0).sum())
        n_total  = seg.size
        print(f"mean_out={mean_out:.4f}  mean_in={mean_in:.4f}  "
              f"inside={n_inside}/{n_total} ({100*n_inside/n_total:.1f}%)")
        stats.append(dict(name=czi_path.name, mean_out=mean_out, mean_in=mean_in,
                          inside_frac=n_inside/n_total))

    for ax, title in zip(axes, ['All pixels', 'Inside-cell pixels only']):
        ax.set_xlabel('Normalised intensity\n(img - mean_out) / (mean_in - mean_out)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.4)
        ax.axvline(1, color='k', lw=0.8, ls=':', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = ax_all.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=10,
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        f'CIO normalisation test — vinc control  '
        f'(binned {TARGET}px, RB r={RB_R}, scale=1, no clip)',
        fontsize=11,
    )
    plt.tight_layout()
    out = OUT_DIR / 'cio_rb_histogram.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved → {out}')

    # summary stats
    print("\n--- stats ---")
    for s in stats:
        print(f"  {s['name'][-30:]:<32}  mean_out={s['mean_out']:.4f}  "
              f"mean_in={s['mean_in']:.4f}  inside={s['inside_frac']*100:.1f}%")


if __name__ == '__main__':
    main()
