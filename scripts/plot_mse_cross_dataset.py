#!/usr/bin/env python3
"""
plot_mse_cross_dataset.py
=========================
Compare per-patch reconstruction MSE across training splits and test datasets.

Plots produced
--------------
1. mse_violin.png              — overall distribution per dataset (train/val/vinc/…)
2. mse_violin_by_condition.png — breakdown by control vs ycomp within each dataset
3. mse_violin_by_fa_type.png   — train/val split further broken down by FA type label
                                 (joined from the annotation CSV)
4. mse_summary.csv             — median / mean / IQR table

Data sources
------------
Training (train / val):
  <train_run>/baseline/latents.csv
        columns: filename, condition_name, split ('train'/'val'), recon_mse, norm_mse

Labels (for FA-type breakdown):
  labelling/paxdata_paxpatch_batch1and2_combined_labels.csv
        columns: crop_img_filename, group, classification

Test datasets (vinc / ppax / pfak / nih3t3):
  <other_pax_dir>/baseline/<ds>/interactive.h5  → meta/csv

Usage
-----
  # overfit model, baseline
  python scripts/plot_mse_cross_dataset.py \\
      --train-run  ae_results/test_run_overfit_20260322 \\
      --apply-dir  ae_results/other_paxillin \\
      --variant    baseline

  # overfit model, semisup_both
  python scripts/plot_mse_cross_dataset.py \\
      --train-run  ae_results/test_run_overfit_20260322 \\
      --apply-dir  ae_results/other_paxillin \\
      --variant    semisup_both

  # CIO-trained model, baseline
  python scripts/plot_mse_cross_dataset.py \\
      --train-run  ae_results/test_run_cio_26041923 \\
      --apply-dir  ae_results/other_paxillin_cio/test_run_cio_26041923 \\
      --variant    baseline

Apply-dir layout (auto-detected):
  {apply_dir}/{variant}/{ds}/interactive.h5       ← overfit pipeline
  {apply_dir}/{variant}/{ds}/latents_newdata.csv  ← CIO pipeline
"""

import argparse
import io
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── colour palettes ───────────────────────────────────────────────────────────
_DS_COLOURS = {
    'train' : '#4878CF',
    'val'   : '#6ACC65',
    'vinc'  : '#D65F5F',
    'ppax'  : '#B47CC7',
    'pfak'  : '#C4AD66',
    'nih3t3': '#77BEDB',
}

# FA type display order and colours
FA_ORDER = [
    'Nascent Adhesion',
    'Focal Complex',
    'Focal Adhesion',
    'Fibrillar Adhesion',
    'No Adhesion',
    'Uncertain',
    'Unlabelled',
]
_FA_COLOURS = {
    'Nascent Adhesion'  : '#E8A838',
    'Focal Complex'     : '#D65F5F',
    'Focal Adhesion'    : '#B47CC7',
    'Fibrillar Adhesion': '#4878CF',
    'No Adhesion'       : '#888888',
    'Uncertain'         : '#BBBBBB',
    'Unlabelled'        : '#DDDDDD',
}

# Normalise raw classification strings → canonical names
_FA_MAP = {
    'Nascent Adhesion'  : 'Nascent Adhesion',
    'focal complex'     : 'Focal Complex',
    'focal adhesion'    : 'Focal Adhesion',
    'fibrillar adhesion': 'Fibrillar Adhesion',
    'No adhesion'       : 'No Adhesion',
    'Uncertain'         : 'Uncertain',
}

DATASET_ORDER  = ['train', 'val', 'vinc', 'ppax', 'pfak', 'nih3t3']
DATASET_LABELS = {
    'train' : 'Train\n(vinc)',
    'val'   : 'Val\n(vinc)',
    'vinc'  : 'Vinc\n(test)',
    'ppax'  : 'pPax118',
    'pfak'  : 'pFAK',
    'nih3t3': 'NIH3T3',
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_training(train_run_dir: Path, variant: str) -> pd.DataFrame:
    df = pd.read_csv(train_run_dir / variant / 'latents.csv')
    df['dataset'] = df['split']
    return df[['dataset', 'filename', 'condition_name', 'recon_mse', 'norm_mse']]


def load_labels(label_csv: Path) -> pd.DataFrame:
    """Load annotation CSV; return (group, crop_img_filename, fa_type)."""
    df = pd.read_csv(label_csv)
    df['fa_type'] = df['classification'].map(_FA_MAP).fillna('Uncertain')
    return df[['group', 'crop_img_filename', 'fa_type']]


def join_labels(df_latents: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """Attach fa_type to latents by stripping the condition prefix from filename.

    latents filename  : 'control_f0001x0592y0560ps32.tif'
    labels crop_img_filename: 'f0001x0592y0560ps32.tif'  (group='control')
    """
    df = df_latents.copy()
    df['bare_filename'] = df.apply(
        lambda r: r['filename'][len(r['condition_name']) + 1:], axis=1
    )
    merged = df.merge(
        df_labels.rename(columns={'group': 'condition_name',
                                  'crop_img_filename': 'bare_filename'}),
        on=['condition_name', 'bare_filename'],
        how='left',
    )
    merged['fa_type'] = merged['fa_type'].fillna('Unlabelled')
    return merged


def load_other_pax(apply_dir: Path, variant: str, ds: str) -> pd.DataFrame:
    """Load test-dataset MSE from either CIO (latents_newdata.csv) or overfit (interactive.h5)."""
    base = apply_dir / variant / ds
    csv_path = base / 'latents_newdata.csv'
    h5_path  = base / 'interactive.h5'

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif h5_path.exists():
        with h5py.File(h5_path, 'r') as f:
            df = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
    else:
        raise FileNotFoundError(
            f'Neither latents_newdata.csv nor interactive.h5 found in {base}'
        )

    df['dataset'] = ds
    return df[['dataset', 'recon_mse', 'norm_mse', 'condition_name']]


# ── shared violin helper ──────────────────────────────────────────────────────

def _draw_violins(ax, groups, values, colours, tick_labels,
                  metric: str, clip_pct: float = 99.5,
                  separator_after: int = None):
    """Draw violins on ax; returns vmax used for clipping."""
    all_vals = np.concatenate([v for v in values if len(v)])
    vmax = np.percentile(all_vals, clip_pct) if len(all_vals) else 1.0

    for i, (v, col) in enumerate(zip(values, colours)):
        if len(v) == 0:
            continue
        vp = ax.violinplot([np.clip(v, 0, vmax)], positions=[i],
                           showmedians=True, showextrema=False, widths=0.7)
        for body in vp['bodies']:
            body.set_facecolor(col)
            body.set_alpha(0.65)
        vp['cmedians'].set_color('black')
        vp['cmedians'].set_linewidth(1.5)
        ax.text(i, vmax * 1.03, f'{np.median(v):.4f}',
                ha='center', va='bottom', fontsize=6.5, color=col)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_ylim(bottom=0, top=vmax * 1.15)
    ax.spines[['top', 'right']].set_visible(False)

    if separator_after is not None and separator_after < len(groups):
        ax.axvline(separator_after - 0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

    return vmax


# ── plot 1: overall per-dataset ───────────────────────────────────────────────

def plot_overall(data_dict: dict, out_dir: Path, variant: str = 'baseline'):
    groups = [k for k in DATASET_ORDER if k in data_dict]
    n_train = sum(1 for g in groups if g in ('train', 'val'))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(groups) * 1.4), 5))
    for ax, metric in zip(axes, ['recon_mse', 'norm_mse']):
        vals   = [data_dict[g][metric].dropna().values for g in groups]
        cols   = [_DS_COLOURS[g] for g in groups]
        labels = [DATASET_LABELS[g] for g in groups]
        _draw_violins(ax, groups, vals, cols, labels, metric,
                      separator_after=n_train)
        ax.set_title(metric, fontsize=11, fontweight='bold')

    fig.suptitle(f'MSE — all datasets  ({variant} model)', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'mse_violin.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')


# ── plot 2: per condition (control / ycomp) ───────────────────────────────────

def plot_by_condition(data_dict: dict, out_dir: Path, variant: str = 'baseline'):
    groups = [k for k in DATASET_ORDER if k in data_dict]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(groups) * 2.4), 5))
    for ax, metric in zip(axes, ['recon_mse', 'norm_mse']):
        pos, tick_pos, tick_labels, colours, values = 0, [], [], [], []
        n_train_cols = 0
        for gi, g in enumerate(groups):
            df_g = data_dict[g]
            for cond in ['control', 'ycomp']:
                sub = df_g[df_g['condition_name'] == cond][metric].dropna().values
                if len(sub) == 0:
                    continue
                col   = _DS_COLOURS[g]
                alpha = 0.75 if cond == 'control' else 0.4
                vp = ax.violinplot([sub], positions=[pos],
                                   showmedians=True, showextrema=False, widths=0.6)
                for body in vp['bodies']:
                    body.set_facecolor(col)
                    body.set_alpha(alpha)
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(1.2)
                tick_pos.append(pos)
                tick_labels.append(f'{DATASET_LABELS[g].split(chr(10))[0]}\n{cond}')
                if g in ('train', 'val'):
                    n_train_cols += 1
                pos += 1
            pos += 0.4

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(bottom=0)
        if n_train_cols > 0 and n_train_cols < len(tick_pos):
            ax.axvline(n_train_cols - 0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

    fig.suptitle(f'MSE by condition (control / ycomp)  ({variant} model)', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'mse_violin_by_condition.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')


# ── plot 3: training splits × FA type ────────────────────────────────────────

def plot_by_fa_type(df_train_labelled: pd.DataFrame, out_dir: Path, variant: str = 'baseline'):
    """Violin per (split × FA type) for labelled training patches."""
    splits = ['train', 'val']
    fa_types = [ft for ft in FA_ORDER if ft in df_train_labelled['fa_type'].values]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(fa_types) * 2.4), 5))

    for ax, metric in zip(axes, ['recon_mse', 'norm_mse']):
        pos, tick_pos, tick_labels = 0, [], []
        n_train_cols = 0

        for split in splits:
            df_s = df_train_labelled[df_train_labelled['dataset'] == split]
            col_base = _DS_COLOURS[split]

            for ft in fa_types:
                sub = df_s[df_s['fa_type'] == ft][metric].dropna().values
                if len(sub) == 0:
                    continue

                fa_col = _FA_COLOURS.get(ft, '#999999')
                vp = ax.violinplot([sub], positions=[pos],
                                   showmedians=True, showextrema=False, widths=0.7)
                for body in vp['bodies']:
                    body.set_facecolor(fa_col)
                    body.set_alpha(0.75 if split == 'train' else 0.45)
                vp['cmedians'].set_color('black' if split == 'train' else '#444444')
                vp['cmedians'].set_linewidth(1.4)

                # n annotation
                ax.text(pos, -0.003, f'n={len(sub)}',
                        ha='center', va='top', fontsize=5.5, color='#555555',
                        transform=ax.get_xaxis_transform())

                tick_pos.append(pos)
                short = ft.replace(' Adhesion', '\nAdhesion').replace('No ', 'No\n')
                tick_labels.append(f'{split}\n{short}')
                if split == 'train':
                    n_train_cols += 1
                pos += 1
            pos += 0.5   # gap between train and val

        all_vals = df_train_labelled[metric].dropna().values
        vmax = np.percentile(all_vals, 99.5) if len(all_vals) else 1.0
        ax.set_ylim(bottom=0, top=vmax * 1.15)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6.5)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        if n_train_cols > 0:
            ax.axvline(n_train_cols - 0.5, color='grey', lw=0.8, ls='--', alpha=0.5)

    # Legend: FA type colours
    from matplotlib.patches import Patch
    handles = [Patch(fc=_FA_COLOURS[ft], alpha=0.75, label=ft)
               for ft in fa_types if ft in _FA_COLOURS]
    handles += [Patch(fc=_DS_COLOURS['train'], alpha=0.75, label='train (solid)'),
                Patch(fc=_DS_COLOURS['val'],   alpha=0.45, label='val (faded)')]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 5),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f'MSE by FA type label — train vs val  ({variant} model, labelled patches only)',
                 fontsize=11)
    plt.tight_layout()
    out = out_dir / 'mse_violin_by_fa_type.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {out}')


# ── summary table ─────────────────────────────────────────────────────────────

def save_summary(data_dict: dict, df_labelled: pd.DataFrame, out_dir: Path):
    rows = []
    for g in DATASET_ORDER:
        if g not in data_dict:
            continue
        for metric in ['recon_mse', 'norm_mse']:
            v = data_dict[g][metric].dropna()
            rows.append(dict(group=g, fa_type='(all)', metric=metric,
                             n=len(v), mean=v.mean(), median=v.median(),
                             p25=v.quantile(0.25), p75=v.quantile(0.75)))

    # Per FA type for train / val
    for split in ['train', 'val']:
        df_s = df_labelled[df_labelled['dataset'] == split]
        for ft in FA_ORDER:
            df_ft = df_s[df_s['fa_type'] == ft]
            if df_ft.empty:
                continue
            for metric in ['recon_mse', 'norm_mse']:
                v = df_ft[metric].dropna()
                rows.append(dict(group=split, fa_type=ft, metric=metric,
                                 n=len(v), mean=v.mean(), median=v.median(),
                                 p25=v.quantile(0.25), p75=v.quantile(0.75)))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'mse_summary.csv', index=False, float_format='%.6f')
    print(f'  saved: {out_dir}/mse_summary.csv')
    print()
    print(summary.to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-folder', default='/home/lding/lding/fa_data_analysis')
    p.add_argument('--train-run',   default='ae_results/test_run_overfit_20260322')
    p.add_argument('--variant',     default='baseline',
                   help='Model variant subfolder, e.g. baseline, semisup_both')
    p.add_argument('--apply-dir',   default='ae_results/other_paxillin',
                   help='Directory containing applied results; supports both '
                        'overfit (interactive.h5) and CIO (latents_newdata.csv) layouts')
    p.add_argument('--out-dir',     default=None,
                   help='Output dir (default: <apply-dir>/mse_cross_dataset/<variant>)')
    p.add_argument('--label-csv',
                   default='labelling/paxdata_paxpatch_batch1and2_combined_labels.csv')
    p.add_argument('--datasets', nargs='+', default=['vinc', 'ppax', 'pfak', 'nih3t3'])
    return p.parse_args()


def main():
    args      = parse_args()
    root      = Path(args.root_folder)
    apply_dir = root / args.apply_dir
    out_path  = args.out_dir or str(Path(args.apply_dir) / 'mse_cross_dataset' / args.variant)
    out_dir   = root / out_path
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Training run : {root / args.train_run}')
    print(f'Variant      : {args.variant}')
    print(f'Apply dir    : {apply_dir}')
    print(f'Labels       : {root / args.label_csv}')
    print(f'Output       : {out_dir}')
    print()

    # ── load training ──────────────────────────────────────────────────────────
    print('Loading training latents …')
    df_latents = load_training(root / args.train_run, args.variant)
    print(f'  {len(df_latents):,} patches  '
          f'(train={( df_latents["dataset"]=="train").sum():,}  '
          f'val={(df_latents["dataset"]=="val").sum():,})')

    # ── join labels ────────────────────────────────────────────────────────────
    print('Joining FA type labels …')
    df_labels = load_labels(root / args.label_csv)
    df_labelled = join_labels(df_latents, df_labels)
    n_lab = (df_labelled['fa_type'] != 'Unlabelled').sum()
    print(f'  {n_lab:,} / {len(df_labelled):,} patches have an FA label')
    print('  FA type breakdown:')
    print(df_labelled.groupby(['dataset', 'fa_type']).size()
          .rename('n').reset_index().to_string(index=False))
    print()

    # ── load test datasets ────────────────────────────────────────────────────
    data_dict = {
        'train': df_labelled[df_labelled['dataset'] == 'train'],
        'val'  : df_labelled[df_labelled['dataset'] == 'val'],
    }
    for ds in args.datasets:
        base = apply_dir / args.variant / ds
        if not (base / 'latents_newdata.csv').exists() and not (base / 'interactive.h5').exists():
            print(f'  [SKIP] {ds}: no latents_newdata.csv or interactive.h5 in {base}')
            continue
        print(f'Loading {ds} …')
        data_dict[ds] = load_other_pax(apply_dir, args.variant, ds)
        print(f'  {ds}: {len(data_dict[ds]):,} patches')

    print()
    print('Generating plots …')
    plot_overall(data_dict, out_dir, args.variant)
    plot_by_condition(data_dict, out_dir, args.variant)
    plot_by_fa_type(df_labelled, out_dir, args.variant)
    save_summary(data_dict, df_labelled, out_dir)
    print()
    print(f'Done → {out_dir}')


if __name__ == '__main__':
    main()
