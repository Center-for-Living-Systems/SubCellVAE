"""
Plot per-source-image pixel intensity histograms from a patch directory.

Each patch (inside-cell crop) contributes all its pixels to the histogram
for the source image it came from (identified by the f#### token in the
filename).  Produces two figures:

  1. overlaid_histograms.png  — all source images overlaid (alpha=0.4),
                                plus a bold overall histogram
  2. grid_histograms.png      — one subplot per source image

Usage:
  python scripts/plot_patch_intensity_histogram.py <patch_dir> [patch_dir2 ...]
  python scripts/plot_patch_intensity_histogram.py \\
      ae_results/pax_ch_patch/patch_control_cell_insideoutside \\
      ae_results/pax_ch_patch/patch_control_cio_norb \\
      --out-dir /tmp/histograms

Tip — compare with/without rolling ball by passing both dirs:
  python scripts/plot_patch_intensity_histogram.py \\
      .../patch_control_cell_insideoutside \\
      .../patch_control_cio_norb
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile

_FNUM_RE = re.compile(r'_f(\d+)')   # matches f0000 in filename stem


def _make_unique_labels(paths: list[Path]) -> list[str]:
    """Return a label per path using the minimum number of path components
    needed so that all labels are unique.  Starts with just the directory
    name; if collisions exist, prepends the parent name, and so on."""
    parts = [list(reversed(p.parts)) for p in paths]   # innermost first
    depth = 1
    while True:
        labels = ['_'.join(reversed(p[:depth])) for p in parts]
        if len(set(labels)) == len(labels) or depth >= max(len(p) for p in parts):
            return labels
        depth += 1


def _load_pixels_by_image(patch_dir: Path) -> dict[str, np.ndarray]:
    """Return {image_id: flat float32 pixel array} for all patches in dir."""
    tifs = sorted(patch_dir.glob('*.tif'))
    if not tifs:
        print(f"  [WARN] no .tif files in {patch_dir}", file=sys.stderr)
        return {}

    by_image: dict[str, list] = defaultdict(list)
    for p in tifs:
        m = _FNUM_RE.search(p.stem)
        img_id = m.group(1) if m else 'unknown'
        arr = tifffile.imread(str(p)).astype(np.float32).ravel()
        by_image[img_id].append(arr)

    return {k: np.concatenate(v) for k, v in sorted(by_image.items())}


def _plot_overlaid(pixels_by_image: dict[str, np.ndarray],
                   label: str, out_path: Path,
                   bins: int = 120, xlim: tuple | None = None) -> None:
    """All source images overlaid + overall bold histogram."""
    all_pixels = np.concatenate(list(pixels_by_image.values()))
    finite     = all_pixels[np.isfinite(all_pixels)]

    if xlim is None:
        lo = np.percentile(finite, 0.5)
        hi = np.percentile(finite, 99.5)
        xlim = (lo, hi)

    edges = np.linspace(xlim[0], xlim[1], bins + 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Per-image histograms (density, overlaid)
    cmap = plt.cm.get_cmap('tab20', len(pixels_by_image))
    for i, (img_id, pix) in enumerate(pixels_by_image.items()):
        pix = pix[np.isfinite(pix)]
        counts, _ = np.histogram(pix, bins=edges, density=True)
        ax.plot(edges[:-1], counts, color=cmap(i), alpha=0.45,
                linewidth=1.0, label=f'img {img_id}')

    # Overall histogram (bold)
    counts_all, _ = np.histogram(finite, bins=edges, density=True)
    ax.plot(edges[:-1], counts_all, color='black', linewidth=2.0,
            label='all patches')

    ax.set_xlim(xlim)
    ax.set_xlabel('Pixel intensity (normalised)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Inside-cell patch intensities — {label}', fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    # Legend only if not too many images
    if len(pixels_by_image) <= 16:
        ax.legend(fontsize=7, ncol=2, loc='upper right', framealpha=0.6)
    else:
        ax.legend(handles=[ax.lines[-1]], labels=['all patches'],
                  fontsize=8, loc='upper right')

    # Vertical reference lines at 0 and 1
    for xv, ls, col in [(0, '--', '#888888'), (1, '-', '#cc3333')]:
        ax.axvline(xv, linestyle=ls, color=col, linewidth=1.2, alpha=0.7)
    ax.text(1.01, ax.get_ylim()[1] * 0.95, 'x=1', color='#cc3333',
            fontsize=8, va='top')

    # Summary stats
    p50, p95, p99 = np.percentile(finite, [50, 95, 99])
    stats = (f'n_imgs={len(pixels_by_image)}  n_px={len(finite):,}\n'
             f'median={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}  max={finite.max():.3f}')
    ax.text(0.02, 0.97, stats, transform=ax.transAxes,
            fontsize=8, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_grid(pixels_by_image: dict[str, np.ndarray],
               label: str, out_path: Path,
               bins: int = 80, xlim: tuple | None = None) -> None:
    """One subplot per source image."""
    n = len(pixels_by_image)
    if n == 0:
        return

    all_pixels = np.concatenate(list(pixels_by_image.values()))
    finite_all = all_pixels[np.isfinite(all_pixels)]
    if xlim is None:
        lo = np.percentile(finite_all, 0.5)
        hi = np.percentile(finite_all, 99.5)
        xlim = (lo, hi)
    edges = np.linspace(xlim[0], xlim[1], bins + 1)

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3, nrows * 2.2),
                              sharex=True, sharey=False)
    axes = np.array(axes).ravel()

    cmap = plt.cm.get_cmap('tab20', n)
    for i, (img_id, pix) in enumerate(pixels_by_image.items()):
        ax = axes[i]
        pix = pix[np.isfinite(pix)]
        counts, _ = np.histogram(pix, bins=edges, density=True)
        ax.fill_between(edges[:-1], counts, alpha=0.6, color=cmap(i), step='post')
        ax.axvline(1.0, color='#cc3333', linewidth=1.0, alpha=0.7, linestyle='-')
        ax.axvline(0.0, color='#888888', linewidth=0.8, linestyle='--')
        p99 = np.percentile(pix, 99)
        ax.set_title(f'img {img_id}\np99={p99:.2f}', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Per-image inside-cell intensities — {label}', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _plot_compare(dirs_pixels: list[tuple[str, np.ndarray]],
                  out_path: Path,
                  bins: int = 120, xlim: tuple | None = None) -> None:
    """One figure with one bold overall-histogram line per directory."""
    all_px = np.concatenate([px for _, px in dirs_pixels])
    finite_all = all_px[np.isfinite(all_px)]

    if xlim is None:
        lo = np.percentile(finite_all, 0.5)
        hi = np.percentile(finite_all, 99.5)
        xlim = (lo, hi)

    edges = np.linspace(xlim[0], xlim[1], bins + 1)

    cmap = plt.cm.get_cmap('tab10', len(dirs_pixels))
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (label, px) in enumerate(dirs_pixels):
        px = px[np.isfinite(px)]
        counts, _ = np.histogram(px, bins=edges, density=True)
        p50, p95, p99 = np.percentile(px, [50, 95, 99])
        lbl = f'{label}  (med={p50:.3f} p95={p95:.3f} p99={p99:.3f} max={px.max():.3f})'
        ax.plot(edges[:-1], counts, color=cmap(i), linewidth=2.0, label=lbl)

    for xv, ls, col in [(0, '--', '#888888'), (1, '-', '#cc3333')]:
        ax.axvline(xv, linestyle=ls, color=col, linewidth=1.2, alpha=0.7)

    ax.set_xlim(xlim)
    ax.set_xlabel('Pixel intensity (normalised)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Patch intensity comparison', fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('patch_dirs', nargs='+',
                    help='One or more patch directories to plot')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: each patch_dir/histograms/)')
    ap.add_argument('--bins', type=int, default=120,
                    help='Number of histogram bins (default: 120)')
    ap.add_argument('--xlim', nargs=2, type=float, default=None,
                    metavar=('LO', 'HI'),
                    help='Fixed x-axis range, e.g. --xlim 0 1.2')
    ap.add_argument('--compare', action='store_true',
                    help='Overlay all directories on one comparison figure')
    args = ap.parse_args()

    xlim = tuple(args.xlim) if args.xlim else None

    valid_dirs = [Path(p) for p in args.patch_dirs if Path(p).is_dir()]
    for p in args.patch_dirs:
        if not Path(p).is_dir():
            print(f"[WARN] Not a directory: {p}", file=sys.stderr)
    labels = _make_unique_labels(valid_dirs)

    if args.compare:
        dirs_pixels = []
        for patch_dir, label in zip(valid_dirs, labels):
            print(f"\n[histogram] loading {label}")
            pixels_by_image = _load_pixels_by_image(patch_dir)
            if not pixels_by_image:
                continue
            all_px = np.concatenate(list(pixels_by_image.values()))
            n_patches = sum(len(v) // (32 * 32) for v in pixels_by_image.values())
            print(f"  {len(pixels_by_image)} source images, ~{n_patches} patches")
            dirs_pixels.append((label, all_px))

        if dirs_pixels:
            out_dir = Path(args.out_dir) if args.out_dir else valid_dirs[0] / 'histograms'
            out_dir.mkdir(parents=True, exist_ok=True)
            _plot_compare(dirs_pixels, out_dir / 'comparison.png',
                          bins=args.bins, xlim=xlim)
    else:
        for patch_dir, label in zip(valid_dirs, labels):
            out_dir = Path(args.out_dir) if args.out_dir else patch_dir / 'histograms'
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[histogram] {label}")
            pixels_by_image = _load_pixels_by_image(patch_dir)
            if not pixels_by_image:
                continue

            n_images  = len(pixels_by_image)
            n_patches = sum(len(v) // (32 * 32) for v in pixels_by_image.values())
            print(f"  {n_images} source images, ~{n_patches} patches")

            _plot_overlaid(pixels_by_image, label,
                           out_dir / f'overlaid_{label}.png',
                           bins=args.bins, xlim=xlim)
            _plot_grid(pixels_by_image, label,
                       out_dir / f'grid_{label}.png',
                       bins=args.bins, xlim=xlim)

    print("\n[histogram] Done")


if __name__ == '__main__':
    main()
