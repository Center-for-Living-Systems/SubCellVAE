"""
Pack a (variant, dataset) result directory into a structured HDF5
suitable for the interactive patch viewer (view_interactive.py).

Reads:
  {result_dir}/latents_newdata.csv
  {result_dir}/analysis/analysis_results.csv   (UMAP_1, UMAP_2)
  {result_dir}/fa_cls_lat8/predictions_all.csv
  {result_dir}/pos_cls_lat8/predictions_all.csv
  {result_dir}/recon/patches_raw.tif  + patches_index.csv
  {result_dir}/recon/patches_recon.tif
  {result_dir}/recon/images_raw.tif   + images_index.csv

Writes:
  {result_dir}/interactive.h5   (or --out path)

Usage:
    python scripts/pack_interactive_h5.py <result_dir>
    python scripts/pack_interactive_h5.py <result_dir> --out /tmp/my.h5
    python scripts/pack_interactive_h5.py <result_dir> --image-scale 0.5
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

# Regex matching the patch filename pattern:
#   {group}_f{img_id}x{x_c}y{y_c}ps{ps}[.tif]
_COORD_RE = re.compile(r'^(.+_f\d+)x(\d+)y(\d+)ps(\d+)')


def _parse_patch_coords(filename: str):
    """Return (group, x_c, y_c, ps) parsed from the patch filename stem.

    Coordinates (x_c, y_c) are in the *padded* image space; subtract
    pad_size from each to get the canvas coordinates used by the viewer.
    Returns (None, None, None, None) on mismatch.
    """
    stem = Path(str(filename)).stem
    m = _COORD_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None, None, None, None


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def _encode_patch_b64(arr_f32: np.ndarray, zoom: int = 4) -> str:
    """Encode a (H, W) float32 [0,1] array as a base64 PNG string.

    The patch is zoomed by `zoom` using nearest-neighbour so it renders
    clearly in the Bokeh hover tooltip at ~128 px.
    """
    img = _to_uint8(arr_f32)
    if zoom > 1:
        img = img.repeat(zoom, axis=0).repeat(zoom, axis=1)
    buf = io.BytesIO()
    Image.fromarray(img, mode='L').save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result_dir',
                    help='Result directory at the variant/dataset level '
                         '(e.g. ae_results/other_paxillin/baseline/vinc)')
    ap.add_argument('--out', default=None,
                    help='Output HDF5 path (default: result_dir/interactive.h5)')
    ap.add_argument('--pad-size', type=int, default=64,
                    help='Pad size used during patch extraction (default: 64)')
    ap.add_argument('--image-scale', type=float, default=1.0,
                    help='Downscale full canvas images by this factor to save space '
                         '(default: 1.0 = no downscale; try 0.5 for large datasets)')
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_h5 = Path(args.out) if args.out else result_dir / 'interactive.h5'
    pad = args.pad_size

    if not result_dir.is_dir():
        print(f"[pack] ERROR: {result_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"[pack] {result_dir}")
    print(f"[pack] → {out_h5}")

    # ── Load and merge metadata CSVs ─────────────────────────────────────────
    latents_csv  = result_dir / 'latents_newdata.csv'
    fa_pred_csv  = result_dir / 'fa_cls_lat8'  / 'predictions_all.csv'
    pos_pred_csv = result_dir / 'pos_cls_lat8' / 'predictions_all.csv'
    analysis_csv = result_dir / 'analysis'     / 'analysis_results.csv'

    if not latents_csv.exists():
        print(f"[pack] ERROR: {latents_csv} not found — run ae_apply first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(latents_csv)
    print(f"[pack]   {len(df)} patches in latents_newdata.csv")

    # Parse patch filename → group, canvas_cx, canvas_cy, ps
    parsed = df['filename'].apply(
        lambda f: pd.Series(
            dict(zip(['patch_group', 'x_c', 'y_c', 'ps'],
                     _parse_patch_coords(f)))
        )
    )
    df = pd.concat([df, parsed], axis=1)
    df['canvas_cx'] = pd.to_numeric(df['x_c'], errors='coerce') - pad
    df['canvas_cy'] = pd.to_numeric(df['y_c'], errors='coerce') - pad

    # FA type predictions
    if fa_pred_csv.exists():
        fa = pd.read_csv(fa_pred_csv)
        proba_cols = [c for c in fa.columns if c.startswith('proba_')]
        fa_sub = fa[['filename', 'pred_label'] + proba_cols].copy()
        fa_sub = fa_sub.rename(columns={'pred_label': 'fa_pred'})
        fa_sub = fa_sub.rename(columns={c: f'fa_{c}' for c in proba_cols})
        df = df.merge(fa_sub, on='filename', how='left')
        print(f"[pack]   FA predictions merged ({fa_sub['fa_pred'].nunique()} classes)")
    else:
        print(f"[pack]   WARN: {fa_pred_csv} not found")

    # Position predictions
    if pos_pred_csv.exists():
        pos = pd.read_csv(pos_pred_csv)
        proba_cols = [c for c in pos.columns if c.startswith('proba_')]
        pos_sub = pos[['filename', 'pred_label'] + proba_cols].copy()
        pos_sub = pos_sub.rename(columns={'pred_label': 'pos_pred'})
        pos_sub = pos_sub.rename(columns={c: f'pos_{c}' for c in proba_cols})
        df = df.merge(pos_sub, on='filename', how='left')
        print(f"[pack]   Position predictions merged ({pos_sub['pos_pred'].nunique()} classes)")
    else:
        print(f"[pack]   WARN: {pos_pred_csv} not found")

    # UMAP coordinates from analysis
    if analysis_csv.exists():
        ana = pd.read_csv(analysis_csv)
        umap_cols = [c for c in ana.columns if c.upper().startswith('UMAP')]
        if len(umap_cols) >= 2:
            rename_map = {umap_cols[0]: 'UMAP_1', umap_cols[1]: 'UMAP_2'}
            ana_sub = ana[['filename'] + umap_cols[:2]].rename(columns=rename_map)
            df = df.merge(ana_sub, on='filename', how='left')
            print(f"[pack]   UMAP coordinates merged")
        else:
            print(f"[pack]   WARN: no UMAP columns found in {analysis_csv}")
    else:
        print(f"[pack]   WARN: {analysis_csv} not found — viewer will fall back to latent dims")

    # ── Load patch stacks ─────────────────────────────────────────────────────
    recon_dir         = result_dir / 'recon'
    patches_raw_tif   = recon_dir  / 'patches_raw.tif'
    patches_recon_tif = recon_dir  / 'patches_recon.tif'
    patches_idx_csv   = recon_dir  / 'patches_index.csv'

    patches_raw_arr = patches_recon_arr = None

    if patches_raw_tif.exists() and patches_idx_csv.exists():
        patch_idx   = pd.read_csv(patches_idx_csv)
        raw_stack   = tifffile.imread(str(patches_raw_tif))
        recon_stack = tifffile.imread(str(patches_recon_tif))
        n_frames, H, W = raw_stack.shape[:3]

        name_to_frame = {row['name']: int(row['frame'])
                         for _, row in patch_idx.iterrows()}
        stems         = df['filename'].apply(lambda f: Path(str(f)).stem)
        frame_indices = stems.map(name_to_frame)

        n = len(df)
        patches_raw_arr   = np.zeros((n, H, W), dtype=np.float32)
        patches_recon_arr = np.zeros((n, H, W), dtype=np.float32)
        for i, frame in enumerate(frame_indices):
            if pd.notna(frame):
                f = int(frame)
                patches_raw_arr[i]   = raw_stack[f]
                patches_recon_arr[i] = recon_stack[f]

        print(f"[pack]   {n} patches loaded from TIFFs ({H}×{W}px)")

        # Encode patches as base64 PNG for Bokeh hover tooltips
        print(f"[pack]   Encoding patches as base64 PNG for hover tooltips …")
        df['raw_b64']   = [_encode_patch_b64(patches_raw_arr[i])   for i in range(n)]
        df['recon_b64'] = [_encode_patch_b64(patches_recon_arr[i]) for i in range(n)]
        print(f"[pack]   Done encoding")
    else:
        print(f"[pack]   WARN: patch TIFFs not found in {recon_dir}")

    # ── Load full canvas images ───────────────────────────────────────────────
    images_raw_tif = recon_dir / 'images_raw.tif'
    images_idx_csv = recon_dir / 'images_index.csv'

    images_raw_arr = None
    img_meta_df    = None

    if images_raw_tif.exists() and images_idx_csv.exists():
        img_meta_df    = pd.read_csv(images_idx_csv)
        images_raw_arr = tifffile.imread(str(images_raw_tif))  # (M, H', W')

        if args.image_scale != 1.0:
            s = args.image_scale
            scaled_list = []
            for i in range(images_raw_arr.shape[0]):
                img = images_raw_arr[i]
                pil = (Image.fromarray(_to_uint8(img), mode='L')
                       if img.ndim == 2
                       else Image.fromarray(_to_uint8(img)))
                new_sz = (max(1, int(pil.width * s)), max(1, int(pil.height * s)))
                scaled_list.append(np.array(pil.resize(new_sz, Image.LANCZOS),
                                            dtype=np.uint8))
            images_raw_arr = np.stack(scaled_list)
        else:
            # Keep as float32 (as loaded from TIFF)
            pass

        print(f"[pack]   {images_raw_arr.shape[0]} canvas images loaded "
              f"({images_raw_arr.shape[-2]}×{images_raw_arr.shape[-1]}px)")
    else:
        print(f"[pack]   WARN: image TIFFs not found in {recon_dir}")

    # ── Write HDF5 ────────────────────────────────────────────────────────────
    print(f"[pack]   Writing HDF5 …")
    with h5py.File(out_h5, 'w') as f:
        # Metadata as UTF-8 CSV string
        f.create_dataset('meta/csv', data=df.to_csv(index=False).encode('utf-8'))

        f.attrs['pad_size']    = pad
        f.attrs['image_scale'] = args.image_scale
        f.attrs['result_dir']  = str(result_dir)
        f.attrs['n_patches']   = len(df)

        if patches_raw_arr is not None:
            f.create_dataset('patches/raw',   data=patches_raw_arr,
                             compression='gzip', compression_opts=4)
            f.create_dataset('patches/recon', data=patches_recon_arr,
                             compression='gzip', compression_opts=4)

        if images_raw_arr is not None:
            f.create_dataset('images/raw',  data=images_raw_arr,
                             compression='gzip', compression_opts=4)
            f.create_dataset('images/meta',
                             data=img_meta_df.to_csv(index=False).encode('utf-8'))

    size_mb = out_h5.stat().st_size / 1e6
    print(f"[pack]   Done — {out_h5.name}  ({size_mb:.1f} MB,  {len(df)} patches)")


if __name__ == '__main__':
    main()
