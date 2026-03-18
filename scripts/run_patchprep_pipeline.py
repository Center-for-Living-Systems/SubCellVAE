"""
run_patchprep_pipeline.py
=========================
Command-line wrapper for ``patchprep_pipeline.run_pipeline()``.

Typical usage
-------------
Run with all defaults (reads hard-coded paths + settings below):

    python run_patchprep_pipeline.py

Override individual settings from the command line:

    python run_patchprep_pipeline.py --condition ctrl --start_ind 0 --end_ind 10
    python run_patchprep_pipeline.py --condition y    --patch_size 64 --mask_ratio 0.3
    python run_patchprep_pipeline.py --rand_trans --rand_rota --max_shift_px 8 --max_angle_deg 30
    python run_patchprep_pipeline.py --debug

Dry-run (prints resolved config, does not process any files):

    python run_patchprep_pipeline.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the repo root (parent of this script's directory) is on sys.path
# so that `utils.patch_prep` can be imported.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from patchprep_pipeline import PipelineConfig, run_pipeline   # noqa: E402


# ===========================================================================
# ✏️  EDIT THIS SECTION to match your project layout
# ===========================================================================

DATA_ROOT    = Path("/mnt/d/lding/FA/data/FA_ML_Annabel_20250217/031125data")
RESULTS_ROOT = Path("/mnt/d/lding/FA/analysis_results/FA_ML_Annabel_20250217/031125")

SEG_FOLDER_STR = "code_org_20250820_seg"   # sub-folder inside RESULTS_ROOT/<ctrl_y_str>/

# Map condition label → raw image sub-directory
IMAGE_FOLDER_MAP: dict[str, Path] = {
    "ctrl": DATA_ROOT / "Control",
    "y":    DATA_ROOT / "Ycomp",
}

# Default experimental settings
DEFAULT_CONDITION  = "y"
DEFAULT_MAJOR_CH   = 1
DEFAULT_PATCH_SIZE = 32
DEFAULT_MASK_RATIO = 0.4
DEFAULT_START_IND  = 0
DEFAULT_END_IND    = 5

# ===========================================================================


def _build_derived_paths(
    condition: str,
    major_ch: int,
    patch_size: int,
    time_str: str,
) -> tuple[Path, Path, Path, Path]:
    """Return (image_folder, cell_mask_folder, patch_dir, plot_dir)."""
    ctrl_y_str   = f"{condition}_ch{major_ch}_major"
    group_prefix = f"{condition}_ch{major_ch}"
    group_str    = f"{group_prefix}_patches_gridonly_wholecell_pslocation00"

    if condition not in IMAGE_FOLDER_MAP:
        raise ValueError(
            f"condition must be one of {list(IMAGE_FOLDER_MAP)}, got {condition!r}"
        )
    image_folder     = IMAGE_FOLDER_MAP[condition]
    cell_mask_folder = RESULTS_ROOT / ctrl_y_str / SEG_FOLDER_STR / "mask"

    patch_dir = (
        RESULTS_ROOT / ctrl_y_str / group_str
        / f"tiff_patches{patch_size}_40p_{time_str}"
    )
    plot_dir = (
        RESULTS_ROOT / ctrl_y_str / group_str
        / f"plot_patches{patch_size}_40p_{time_str}"
    )
    return image_folder, cell_mask_folder, patch_dir, plot_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract image patches from .czi microscopy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- experiment ---
    exp = p.add_argument_group("Experiment")
    exp.add_argument(
        "--condition", default=DEFAULT_CONDITION,
        choices=list(IMAGE_FOLDER_MAP),
        help="Experimental condition (selects image sub-directory).",
    )
    exp.add_argument(
        "--major_ch", type=int, default=DEFAULT_MAJOR_CH,
        help="Channel index to extract from .czi (0-based).",
    )

    # --- patch geometry ---
    geom = p.add_argument_group("Patch geometry")
    geom.add_argument(
        "--patch_size", type=int, default=DEFAULT_PATCH_SIZE,
        help="Side length (px) of the final square patch.",
    )
    geom.add_argument(
        "--mask_ratio", type=float, default=DEFAULT_MASK_RATIO,
        help="Minimum mean mask coverage required to keep a patch.",
    )

    # --- file range ---
    rng = p.add_argument_group("File range")
    rng.add_argument(
        "--start_ind", type=int, default=DEFAULT_START_IND,
        help="First file index to process (inclusive).",
    )
    rng.add_argument(
        "--end_ind", type=int, default=DEFAULT_END_IND,
        help="Last file index to process (exclusive).",
    )

    # --- augmentation ---
    aug = p.add_argument_group("Augmentation")
    aug.add_argument(
        "--rand_trans", action="store_true",
        help="Enable random translation augmentation.",
    )
    aug.add_argument(
        "--max_shift_px", type=int, default=0,
        help="Maximum translation in pixels (requires --rand_trans).",
    )
    aug.add_argument(
        "--rand_rota", action="store_true",
        help="Enable random rotation augmentation.",
    )
    aug.add_argument(
        "--max_angle_deg", type=float, default=0.0,
        help="Maximum rotation angle in degrees (requires --rand_rota).",
    )

    # --- normalization ---
    norm = p.add_argument_group("Normalization")
    norm.add_argument(
        "--norm_mode",
        default=None,
        choices=["dataset", "image"],
        help=(
            "Pixel intensity normalization mode applied to each loaded image. "
            "'dataset': whole-dataset per-channel 1%%–99%% percentile stretch "
            "(stats collected across all files before the main loop; use "
            "--norm_channels to specify which channels to include). "
            "'image': per-image per-channel 1%%–99%% stretch computed on the fly. "
            "Omit to disable normalization (raw /255² scaling)."
        ),
    )
    norm.add_argument(
        "--norm_channels",
        type=int,
        nargs="+",
        default=None,
        metavar="CH",
        help=(
            "Space-separated channel indices to include when computing "
            "dataset-level stats (e.g. --norm_channels 0 1 2 3). "
            "Required for --norm_mode dataset; defaults to major_ch only."
        ),
    )
    norm.add_argument(
        "--norm_lo", type=float, default=1.0,
        help="Lower percentile bound for normalization (default: 1.0).",
    )
    norm.add_argument(
        "--norm_hi", type=float, default=99.0,
        help="Upper percentile bound for normalization (default: 99.0).",
    )

    # --- path overrides ---
    paths = p.add_argument_group("Path overrides (optional)")
    paths.add_argument(
        "--image_folder",
        help="Override image directory (default: derived from --condition).",
    )
    paths.add_argument(
        "--cell_mask_folder",
        help="Override cell-mask directory.",
    )
    paths.add_argument(
        "--patch_output_dir",
        help="Override patch .tif output directory.",
    )
    paths.add_argument(
        "--plot_output_dir",
        help="Override plot / CSV output directory.",
    )

    # --- misc ---
    misc = p.add_argument_group("Misc")
    misc.add_argument(
        "--file_type",
        default="czi",
        choices=["czi", "npy"],
        help=(
            "Input image file format. "
            "'czi': Zeiss .czi files (values divided by 255²). "
            "'npy': NumPy .npy files (values used as-is). "
            "Default: czi."
        ),
    )
    misc.add_argument(
        "--pad_size", type=int, default=64,
        help="Padding added to each edge of the loaded image / mask.",
    )
    misc.add_argument(
        "--dpi", type=int, default=256,
        help="DPI for accumulation-plot figures.",
    )
    misc.add_argument(
        "--debug", action="store_true",
        help="Break after first grid position per file (quick sanity-check).",
    )
    misc.add_argument(
        "--dry_run", action="store_true",
        help="Print resolved configuration and exit without processing.",
    )
    misc.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_logging(args.log_level)

    time_str = datetime.now().strftime("%Y%m%d_%H%M")

    # ---- resolve paths ----
    image_folder, cell_mask_folder, patch_dir, plot_dir = _build_derived_paths(
        args.condition, args.major_ch, args.patch_size, time_str
    )

    # apply any explicit overrides
    if args.image_folder:
        image_folder = Path(args.image_folder)
    if args.cell_mask_folder:
        cell_mask_folder = Path(args.cell_mask_folder)
    if args.patch_output_dir:
        patch_dir = Path(args.patch_output_dir)
    if args.plot_output_dir:
        plot_dir = Path(args.plot_output_dir)

    # ---- build config ----
    cfg = PipelineConfig(
        image_folder=image_folder,
        cell_mask_folder=cell_mask_folder,
        movie_partitioned_data_dir=patch_dir,
        movie_plot_dir=plot_dir,
        condition=args.condition,
        major_ch=args.major_ch,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        start_ind=args.start_ind,
        end_ind=args.end_ind,
        rand_trans_flag=args.rand_trans,
        rand_rota_flag=args.rand_rota,
        max_shift_px=args.max_shift_px,
        max_angle_deg=args.max_angle_deg,
        debug_flag=args.debug,
        pad_size=args.pad_size,
        dpi=args.dpi,
        norm_mode=args.norm_mode,
        norm_channels=args.norm_channels,
        norm_lo=args.norm_lo,
        norm_hi=args.norm_hi,
        file_type=args.file_type,
    )

    # ---- dry run ----
    if args.dry_run:
        print("\n=== DRY RUN – resolved configuration ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<35} {v}")
        print("\nNo files processed. Remove --dry_run to run for real.")
        return

    # ---- run ----
    record = run_pipeline(cfg)

    # ---- summary ----
    logging.getLogger(__name__).info(
        "Done. %d total patches accepted across files %d–%d.",
        len(record), args.start_ind, min(args.end_ind, args.end_ind) - 1,
    )


if __name__ == "__main__":
    main()