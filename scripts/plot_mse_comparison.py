"""
Plot MSE comparison across training splits and all other-paxillin test datasets.

For each AE variant, produces:
  1. mse_comparison.png   — bar plot (mean ± std) across train / val /
                            vinc_ctrl / vinc_ycomp / pfak_ctrl / ... (all 10 groups)
  2. mse_by_fa_type.png   — same bars but split further by FA-type label
                            (only for variants that have annotation labels)

Sources:
  training split  → {train_run_dir}/{variant}/latents.csv
                     columns: recon_mse, split (train/val), annotation_label_name (optional)
  test datasets   → {apply_dir}/{variant}/{dataset}/latents_newdata.csv
                     columns: recon_mse, condition_name

Usage:
  python scripts/plot_mse_comparison.py \\
      --train-run-dir /path/to/ae_results/test_run_cio_XXXXXX \\
      --apply-dir     /path/to/ae_results/other_paxillin_cio/test_run_cio_XXXXXX \\
      --out-dir       /path/to/ae_results/other_paxillin_cio/test_run_cio_XXXXXX/mse_comparison

  # To run for specific variants only:
  python scripts/plot_mse_comparison.py \\
      --train-run-dir ... --apply-dir ... --out-dir ... \\
      --variants baseline conae semisup_fa semicon_fa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_VARIANTS = [
    'baseline', 'semisup_fa', 'semisup_pos', 'semisup_both',
    'conae', 'semicon_fa', 'semicon_pos', 'semicon_both',
]
DATASETS = ['vinc', 'pfak', 'ppax', 'nih3t3']
CONDITIONS = ['control', 'ycomp']

FA_ORDER = [
    'Nascent Adhesion', 'focal complex', 'focal adhesion',
    'fibrillar adhesion', 'No adhesion',
]

# Group display order and labels for the main comparison plot
TRAIN_GROUPS = ['train', 'val']
TEST_GROUPS  = [f'{ds}_{cond}' for ds in DATASETS for cond in CONDITIONS]
ALL_GROUPS   = TRAIN_GROUPS + TEST_GROUPS

GROUP_LABELS = {
    'train'        : 'Train',
    'val'          : 'Val',
    'vinc_control' : 'vinc\nctrl',
    'vinc_ycomp'   : 'vinc\nycomp',
    'pfak_control' : 'pfak\nctrl',
    'pfak_ycomp'   : 'pfak\nycomp',
    'ppax_control' : 'ppax\nctrl',
    'ppax_ycomp'   : 'ppax\nycomp',
    'nih3t3_control': 'nih3t3\nctrl',
    'nih3t3_ycomp' : 'nih3t3\nycomp',
}

TRAIN_COLOR = '#4878CF'
VAL_COLOR   = '#6ACC65'
TEST_COLOR  = '#D65F5F'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training(train_run_dir: Path, variant: str) -> pd.DataFrame | None:
    """Load latents.csv from training run; add 'group' column (train/val)."""
    csv = train_run_dir / variant / 'latents.csv'
    if not csv.exists():
        print(f"  [WARN] {csv} not found — skipping training split", file=sys.stderr)
        return None
    df = pd.read_csv(csv)
    if 'split' not in df.columns:
        print(f"  [WARN] no 'split' column in {csv}", file=sys.stderr)
        return None
    df['group'] = df['split'].str.lower()   # 'train' | 'val'
    return df[['group', 'recon_mse'] + (['annotation_label_name']
               if 'annotation_label_name' in df.columns else [])].copy()


def _load_apply(apply_dir: Path, variant: str) -> pd.DataFrame:
    """Load latents_newdata.csv for all datasets; add 'group' column."""
    frames = []
    for ds in DATASETS:
        for cond in CONDITIONS:
            csv = apply_dir / variant / ds / 'latents_newdata.csv'
            if not csv.exists():
                print(f"  [WARN] {csv} not found — skipping", file=sys.stderr)
                continue
            df = pd.read_csv(csv)
            df = df[df['condition_name'] == cond].copy()
            df['group'] = f'{ds}_{cond}'
            keep = ['group', 'recon_mse']
            if 'annotation_label_name' in df.columns:
                keep.append('annotation_label_name')
            frames.append(df[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _collect(train_run_dir: Path, apply_dir: Path, variant: str) -> pd.DataFrame:
    parts = []
    tr = _load_training(train_run_dir, variant)
    if tr is not None:
        parts.append(tr)
    ap = _load_apply(apply_dir, variant)
    if not ap.empty:
        parts.append(ap)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df[np.isfinite(df['recon_mse'])]
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_color(group: str) -> str:
    if group == 'train':
        return TRAIN_COLOR
    if group == 'val':
        return VAL_COLOR
    return TEST_COLOR


def _plot_mse_bars(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Bar plot (mean ± std) of recon_mse across groups."""
    groups = [g for g in ALL_GROUPS if g in df['group'].unique()]
    if not groups:
        return

    means = [df.loc[df['group'] == g, 'recon_mse'].mean() for g in groups]
    stds  = [df.loc[df['group'] == g, 'recon_mse'].std()  for g in groups]
    ns    = [df.loc[df['group'] == g, 'recon_mse'].count() for g in groups]
    sems  = [s / np.sqrt(n) if n > 1 else 0.0 for s, n in zip(stds, ns)]
    labels = [GROUP_LABELS.get(g, g) for g in groups]
    colors = [_bar_color(g) for g in groups]

    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.9), 4.5))
    xs = np.arange(len(groups))
    bars = ax.bar(xs, means, yerr=sems, capsize=4,
                  color=colors, edgecolor='white', linewidth=0.5,
                  error_kw=dict(elinewidth=1.2, ecolor='#333333'))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Reconstruction MSE (mean ± SEM)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Patch count annotations
    for x, n, m in zip(xs, ns, means):
        ax.text(x, m * 0.02, f'n={n}', ha='center', va='bottom',
                fontsize=6, color='#555555')

    # Legend for colour coding
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=TRAIN_COLOR, label='Train'),
        Patch(facecolor=VAL_COLOR,   label='Val'),
        Patch(facecolor=TEST_COLOR,  label='Test datasets'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper right',
              framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_mse_by_fa_type(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Grouped bar plot: x=group, hue=FA type label."""
    if 'annotation_label_name' not in df.columns:
        return
    df = df.dropna(subset=['annotation_label_name'])
    if df.empty:
        return

    fa_classes = [c for c in FA_ORDER if c in df['annotation_label_name'].unique()]
    groups     = [g for g in ALL_GROUPS if g in df['group'].unique()]
    if not fa_classes or not groups:
        return

    n_groups  = len(groups)
    n_classes = len(fa_classes)
    bar_w     = 0.8 / n_classes
    xs        = np.arange(n_groups)

    cmap   = plt.cm.get_cmap('tab10', n_classes)
    colors = [cmap(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), 5))

    for i, fa in enumerate(fa_classes):
        sub = df[df['annotation_label_name'] == fa]
        means, sems = [], []
        for g in groups:
            vals = sub.loc[sub['group'] == g, 'recon_mse'].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        offset = (i - n_classes / 2 + 0.5) * bar_w
        ax.bar(xs + offset, means, yerr=sems, width=bar_w * 0.9,
               capsize=3, color=colors[i], label=fa, alpha=0.85,
               error_kw=dict(elinewidth=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels([GROUP_LABELS.get(g, g) for g in groups], fontsize=9)
    ax.set_ylabel('Reconstruction MSE (mean ± SEM)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7,
              ncol=min(3, n_classes))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--train-run-dir', required=True,
                    help='Training run directory (contains baseline/, conae/, etc.)')
    ap.add_argument('--apply-dir', required=True,
                    help='Apply result directory (contains MODEL_RUN/variant/dataset/)')
    ap.add_argument('--out-dir', required=True,
                    help='Output directory for comparison plots')
    ap.add_argument('--variants', nargs='+', default=ALL_VARIANTS,
                    help='Variants to process (default: all 8)')
    args = ap.parse_args()

    train_run_dir = Path(args.train_run_dir)
    apply_dir     = Path(args.apply_dir)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mse_comparison] train  : {train_run_dir}")
    print(f"[mse_comparison] apply  : {apply_dir}")
    print(f"[mse_comparison] out    : {out_dir}")
    print(f"[mse_comparison] variants: {args.variants}")

    for variant in args.variants:
        print(f"\n── {variant} ──")
        df = _collect(train_run_dir, apply_dir, variant)
        if df.empty:
            print("  No data found — skipping")
            continue

        # 1. Overall MSE bar plot
        _plot_mse_bars(
            df,
            title=f'MSE across datasets  ·  {variant}',
            out_path=out_dir / f'mse_comparison_{variant}.png',
        )

        # 2. MSE by FA type (only if labels present)
        if 'annotation_label_name' in df.columns and df['annotation_label_name'].notna().any():
            _plot_mse_by_fa_type(
                df,
                title=f'MSE by FA type  ·  {variant}',
                out_path=out_dir / f'mse_by_fa_type_{variant}.png',
            )

    print(f"\n[mse_comparison] All done → {out_dir}")


if __name__ == '__main__':
    main()
