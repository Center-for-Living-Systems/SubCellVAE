#!/usr/bin/env python3
"""
Generate config/patchprep_config/ — 32 patchprep configs:
  4 datasets × 2 conditions × 4 normalization modes
    cio        — cell_insideoutside, no rolling ball
    dataset    — percentile stretch,  no rolling ball
    cio_rb     — cell_insideoutside, rolling ball radius 10
    dataset_rb — percentile stretch,  rolling ball radius 10

Output patch path pattern:
  ae_results/patches/{norm}/{dataset}/{condition}/tiff_patches32

Run from repo root:
    python scripts/generate_patchprep_configs.py
"""

from pathlib import Path

ROOT      = '"/home/lding/lding/fa_data_analysis"'
DATA_BASE = "fa_data/other_paxillin"

# ── dataset metadata ─────────────────────────────────────────────────────────
DATASETS = {
    'vinc': {
        'desc'    : 'vinc (20250311 – Vinculin/rb647 + paxillin/m568)  [= training experiment]',
        'subdir'  : '20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568',
        'cond_dir': {'control': 'Control', 'ycomp': 'Ycomp'},
        # Pre-computed masks available for vinc (from original training segmentation).
        # On-the-fly is the default; uncomment cell_mask_folder in config to use them.
        'mask_dir': {
            'control': 'fa_cellseg/031125analysis/control/code_org_20250820_seg/mask',
            'ycomp'  : 'fa_cellseg/031125analysis/ycomp/code_org_20250820_seg/mask',
        },
    },
    'ppax': {
        'desc'    : 'ppax (20250721 – pPaxy118/rb647 + paxillin/m568)  [= previously "newdata"]',
        'subdir'  : '20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568',
        'cond_dir': {'control': 'Control', 'ycomp': 'Y-comp'},
        'mask_dir': None,
    },
    'pfak': {
        'desc'    : 'pfak (20250720 – pFAK/rb647 + paxillin/m568)  [truly new dataset]',
        'subdir'  : '20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025',
        'cond_dir': {'control': 'Control', 'ycomp': 'Ycomp'},
        'mask_dir': None,
    },
    'nih3t3': {
        'desc'    : 'nih3t3 (20260227 – NIH3T3, Vinc/rb647 + pax/m555)  [truly new, different cell line]',
        'subdir'  : '20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH',
        'cond_dir': {'control': 'Control', 'ycomp': 'YCompound'},
        'mask_dir': None,
    },
}

# ── normalization settings ────────────────────────────────────────────────────
_RB_RADIUS = 20

NORM_SETTINGS = {
    'cio': {
        'label'    : 'cell_insideoutside  (no rolling ball)',
        'rb_radius': None,
        'yaml'     : 'normalization:\n  norm_mode : "cell_insideoutside"',
    },
    'dataset': {
        'label'    : 'dataset  (percentile stretch, no rolling ball)',
        'rb_radius': None,
        'yaml'     : ('normalization:\n'
                      '  norm_mode : "dataset"\n'
                      '  norm_lo   : 0.2\n'
                      '  norm_hi   : 99.8'),
    },
    'cio_rb': {
        'label'    : f'cell_insideoutside  (rolling ball radius {_RB_RADIUS})',
        'rb_radius': _RB_RADIUS,
        'yaml'     : 'normalization:\n  norm_mode : "cell_insideoutside"',
    },
    'dataset_rb': {
        'label'    : f'dataset  (percentile stretch, rolling ball radius {_RB_RADIUS})',
        'rb_radius': _RB_RADIUS,
        'yaml'     : ('normalization:\n'
                      '  norm_mode : "dataset"\n'
                      '  norm_lo   : 0.2\n'
                      '  norm_hi   : 99.8'),
    },
}


def make_mask_comment(ds_info: dict, cond: str) -> str:
    """Return commented-out mask_folder line if pre-computed masks exist."""
    if ds_info['mask_dir'] is None:
        return ''
    mask_path = ds_info['mask_dir'][cond]
    return (
        f'  # cell_mask_folder : root_folder + "/{mask_path}"  '
        f'# pre-computed masks (vinc only; uncomment to use instead of on-the-fly)\n'
    )


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for norm, ns in NORM_SETTINGS.items():
        for ds, info in DATASETS.items():
            for cond in ('control', 'ycomp'):
                cond_dir = info['cond_dir'][cond]
                mask_comment = make_mask_comment(info, cond)
                rb = ns['rb_radius']
                rb_line = (f'  rolling_ball_radius : {rb}' if rb is not None
                           else '  rolling_ball_radius : null   # disabled')

                content = f"""\
# Dataset   : {info['desc']}
# Condition : {cond}
# Norm      : {ns['label']}
# Seg       : on-the-fly (default); see commented cell_mask_folder for pre-computed option
paths:
  root_folder      : {ROOT}
  image_folder     : root_folder + "/{DATA_BASE}/{info['subdir']}/{cond_dir}"
  cell_mask_folder : null   # on-the-fly segmentation (default)
{mask_comment}\
  patch_output_dir : root_folder + "/ae_results/patches/{norm}/{ds}/{cond}/tiff_patches32"
  plot_output_dir  : root_folder + "/ae_results/patches/{norm}/{ds}/{cond}/plot_patches32"

experiment:
  condition : "{cond}"
  major_ch  : 1

input:
  file_type : "czi"
  start_ind : 0
  end_ind   : 999

patch:
  patch_size   : 32
  mask_ratio   : 0.4
  pad_size     : 64
  patch_prefix : "{cond}"

preprocessing:
{rb_line}

{ns['yaml']}

segmentation:
  seg_ch                  : 1
  seg_threshold           : 0.1
  seg_close_size          : 11
  seg_min_size_initial    : 3
  seg_min_size_post_close : 10
  seg_min_size_final      : 30000

augmentation:
  rand_trans    : false
  max_shift_px  : 0
  rand_rota     : false
  max_angle_deg : 0.0

misc:
  dpi           : 256
  debug         : false
  log_level     : "INFO"
  use_timestamp : false
"""
                fname = out_dir / f'{ds}_{cond}_{norm}.yaml'
                fname.write_text(content)
                count += 1

    print(f'Generated {count} configs in {out_dir}')


if __name__ == '__main__':
    repo = Path(__file__).parents[1]
    generate(repo / 'config' / 'patchprep_config')
