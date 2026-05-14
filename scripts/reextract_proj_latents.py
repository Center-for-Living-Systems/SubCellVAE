"""
Re-extract latents from trained contrastive model, adding z_proj (p_*) columns to latents.csv.
Usage: python scripts/reextract_proj_latents.py <result_dir>
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

from subcellae.modelling.autoencoders import ContrastiveAE
from subcellae.modelling.dataset import PatchDataset

RESULT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/home/lding/lding/fa_data_analysis/ae_results/contrastive_run/contrastive_cio_rb'
)

print(f"Result dir: {RESULT_DIR}")

df_orig = pd.read_csv(RESULT_DIR / 'latents.csv')
print(f"Original CSV: {len(df_orig)} rows")

# Check if p_ cols already exist
p_existing = [c for c in df_orig.columns if c.startswith('p_')]
if p_existing:
    print(f"p_* columns already present ({len(p_existing)}): {p_existing[0]}...{p_existing[-1]}")
    sys.exit(0)

# Infer model params from existing z_ columns
z_cols = [c for c in df_orig.columns if c.startswith('z_')]
latent_dim = len(z_cols)
print(f"Latent dim: {latent_dim}")

# Load model config to get proj_dim
import yaml
config_candidates = list(RESULT_DIR.glob('*.yaml'))
proj_dim = 64  # default
if config_candidates:
    with open(config_candidates[0]) as f:
        cfg_raw = f.read()
    for line in cfg_raw.splitlines():
        if 'proj_dim' in line and ':' in line:
            try:
                proj_dim = int(line.split(':')[1].strip())
                break
            except ValueError:
                pass
print(f"Proj dim: {proj_dim}")

# Load trained model
ckpt = RESULT_DIR / 'model_final.pt'
loaded = torch.load(ckpt, map_location='cpu', weights_only=False)
if isinstance(loaded, dict):
    model = ContrastiveAE(latent_dim=latent_dim, proj_dim=proj_dim, input_ps=32, no_ch=1, BN_flag=False)
    model.load_state_dict(loaded)
else:
    model = loaded
model.eval()
print(f"Loaded model from {ckpt}")

# Collect unique patch directories from the CSV
patch_dirs_with_meta = {}
for _, row in df_orig.iterrows():
    fp = row['filepath']
    patch_dir = str(Path(fp).parent)
    if patch_dir not in patch_dirs_with_meta:
        patch_dirs_with_meta[patch_dir] = (int(row['condition']), row['condition_name'])

print(f"Patch dirs: {list(patch_dirs_with_meta.keys())}")

datasets = []
for patch_dir, (cond, cond_name) in patch_dirs_with_meta.items():
    ds = PatchDataset(root_dir=patch_dir, condition=cond, condition_name=cond_name)
    datasets.append(ds)
    print(f"  {len(ds)} patches from {patch_dir}")

all_ds = ConcatDataset(datasets)
loader = DataLoader(all_ds, batch_size=256, shuffle=False, num_workers=0)

all_paths, all_z, all_p = [], [], []
with torch.no_grad():
    for i, batch in enumerate(loader):
        x     = batch[0]
        paths = batch[4]
        if x.dim() == 3:      # (B, H, W) → (B, 1, H, W)
            x = x.unsqueeze(1)
        z     = model.encode(x)
        p     = model.project(z)
        all_paths.extend(paths)
        all_z.append(z.cpu().numpy())
        all_p.append(p.cpu().numpy())
        if i % 10 == 0:
            print(f"  Batch {i}: {len(all_paths)} patches processed")

all_z = np.concatenate(all_z, axis=0)
all_p = np.concatenate(all_p, axis=0)
print(f"Extracted z_recon: {all_z.shape}, z_proj: {all_p.shape}")

# Map filepath → projection vector
path_to_p = {p: all_p[i] for i, p in enumerate(all_paths)}

# Add p_* columns
missing = [fp for fp in df_orig['filepath'] if fp not in path_to_p]
if missing:
    print(f"WARNING: {len(missing)} filepaths not found in extracted patches")

for d in range(proj_dim):
    df_orig[f'p_{d}'] = df_orig['filepath'].map(lambda fp: float(path_to_p[fp][d]) if fp in path_to_p else float('nan'))

# Reorder columns
meta_cols   = ['filename', 'filepath', 'condition', 'condition_name', 'group', 'split',
               'recon_mse', 'recon_l1', 'mean_intensity', 'norm_mse', 'recon_hessian_l1']
z_cols_ord  = [f'z_{d}' for d in range(latent_dim)]
p_cols_ord  = [f'p_{d}' for d in range(proj_dim)]
ann_cols    = ['annotation_label', 'annotation_label_name']
extra_cols  = [c for c in df_orig.columns if c not in meta_cols + z_cols_ord + p_cols_ord + ann_cols]
ordered     = meta_cols + z_cols_ord + p_cols_ord + ann_cols + extra_cols
df_orig = df_orig[[c for c in ordered if c in df_orig.columns]]

out = RESULT_DIR / 'latents.csv'
df_orig.to_csv(out, index=False)
print(f"Saved → {out}  ({len(df_orig)} rows, {len(df_orig.columns)} cols)")
print(f"First 25 cols: {list(df_orig.columns[:25])}")
