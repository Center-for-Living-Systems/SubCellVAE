"""
patchprep_pipeline.py
=====================
Core pipeline for extracting image patches from .czi microscopy files.

Provides a single entry-point function `run_pipeline()` that orchestrates:
  1. File discovery
  2. (Optional) Whole-dataset per-channel normalization stat computation
  3. Image + segmentation mask loading / padding + normalization
  4. Grid computation
  5. Patch extraction with optional translation and rotation
  6. Mask-ratio filtering at three levels (big patch, big crop, final crop)
  7. Saving .tif patches and accumulation plots
  8. Writing a CSV record of all accepted patches

Normalization modes (set via ``PipelineConfig.norm_mode``):
  ``None``       – no normalization; raw ``/ 255²`` scaling (original behaviour).
  ``"dataset"``  – whole-dataset, per-channel 1 %–99 % percentile stretch.
                   Stats are collected across ALL files in the run before any
                   patch is extracted, then applied consistently to every image.
  ``"image"``    – per-image, per-channel 1 %–99 % percentile stretch computed
                   on the fly for each loaded image.

All heavy-lifting primitives live in subcellae/dataprep/patch_prep.py; this
module only wires them together.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import subcellae.dataprep.patch_prep as patch_prep

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All tuneable parameters for one pipeline run.

    Parameters
    ----------
    image_folder : str | Path
        Directory that contains the raw .czi files.
    cell_mask_folder : str | Path
        Directory that contains the cell-mask .tif files
        (expected naming: ``cell_mask_<czi_filename>.tif``).
    movie_partitioned_data_dir : str | Path
        Output directory for the extracted .tif patches.
    movie_plot_dir : str | Path
        Output directory for accumulation-plot .png files and the CSV record.
    condition : str
        Short label for the experimental condition (e.g. ``"ctrl"`` or ``"y"``).
        Used only for naming the CSV file.
    major_ch : int
        Channel index to use from the .czi file (0-based). Default ``1``.
    patch_size : int
        Side length (pixels) of the final square patch. Default ``32``.
    mask_ratio : float
        Minimum mean mask value required at each of the three filtering
        stages. Default ``0.4``.
    start_ind : int
        First file index to process (inclusive). Default ``0``.
    end_ind : int
        Last file index to process (exclusive). Default ``5``.
    rand_trans_flag : bool
        Enable random translation augmentation. Default ``False``.
    rand_rota_flag : bool
        Enable random rotation augmentation. Default ``False``.
    max_shift_px : int
        Maximum translation in pixels (used when ``rand_trans_flag`` is True).
        Default ``0``.
    max_angle_deg : float
        Maximum rotation angle in degrees (used when ``rand_rota_flag`` is
        True). Default ``0.0``.
    debug_flag : bool
        If ``True``, break after the first grid position per file (useful for
        quickly checking that everything loads). Default ``False``.
    pad_size : int
        Padding added to each edge of the loaded image / mask. Default ``64``.
    dpi : int
        DPI for accumulation plots. Default ``256``.
    """

    # --- required ---
    image_folder: str | Path
    cell_mask_folder: Optional[str | Path]   # None is valid for on_the_fly mode
    movie_partitioned_data_dir: str | Path
    movie_plot_dir: str | Path
    condition: str

    # --- optional with sensible defaults ---
    major_ch: int = 1
    patch_size: int = 32
    mask_ratio: float = 0.4
    start_ind: int = 0
    end_ind: int = 5

    rand_trans_flag: bool = False
    rand_rota_flag: bool = False
    max_shift_px: int = 0
    max_angle_deg: float = 0.0

    debug_flag: bool = False
    pad_size: int = 64
    dpi: int = 256
    patch_prefix: str = ""           # prepended to patch filenames, e.g. "control"

    # --- normalization ---
    norm_mode: Optional[str] = None
    norm_channels: Optional[list] = None
    norm_lo: float = 1.0
    norm_hi: float = 99.0
    norm_dataset_folder: Optional[str | Path] = None  # if set, compute percentile stats from this folder instead of image_folder
    norm_cell_scale: float = 5.0                      # divisor for "cell_insideoutside" mode

    # --- file type ---
    file_type: str = "czi"

    # --- segmentation ---
    seg_ch: Optional[int] = None          # None → falls back to major_ch
    seg_threshold: float = 0.2
    seg_close_size: int = 5
    seg_min_size_initial: int = 3
    seg_min_size_post_close: int = 10
    seg_min_size_final: int = 30000

    # --- rolling-ball background subtraction ---
    rolling_ball_radius: Optional[float] = None   # None → disabled

    # --- distance-to-boundary features ---
    n_dist_orientations: int = 8          # number of ray directions (produces this many columns)

    # --- derived (computed in __post_init__) ---
    half_ps: int = field(init=False)
    double_ps: int = field(init=False)

    def __post_init__(self):
        self.image_folder = Path(self.image_folder)
        self.movie_partitioned_data_dir = Path(self.movie_partitioned_data_dir)
        self.movie_plot_dir = Path(self.movie_plot_dir)

        self.half_ps = self.patch_size // 2
        self.double_ps = self.patch_size * 2

        if self.cell_mask_folder is not None:
            self.cell_mask_folder = Path(self.cell_mask_folder)

        # validate norm_mode
        _valid_norm_modes = {None, "dataset", "image", "cell_insideoutside"}
        if self.norm_mode not in _valid_norm_modes:
            raise ValueError(
                f"norm_mode must be one of {_valid_norm_modes}, got {self.norm_mode!r}"
            )
        if self.norm_mode == "dataset" and self.norm_channels is None:
            import warnings
            warnings.warn(
                "norm_mode='dataset' but norm_channels is None; "
                f"defaulting to [major_ch={self.major_ch}]. "
                "Pass norm_channels=[0,1,2,3,...] to include all channels.",
                stacklevel=2,
            )
            self.norm_channels = [self.major_ch]

        # validate file_type
        _valid_file_types = {"czi", "npy"}
        if self.file_type not in _valid_file_types:
            raise ValueError(
                f"file_type must be one of {_valid_file_types}, got {self.file_type!r}"
            )

        # ensure output dirs exist
        self.movie_partitioned_data_dir.mkdir(parents=True, exist_ok=True)
        self.movie_plot_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def _process_file(
    filenameID: int,
    filename: str,
    cfg: PipelineConfig,
    norm_stats: Optional[Dict] = None,
) -> list:
    """Process a single .czi file and return a list of record rows.

    Saves patch .tif files and the accumulation plot as side-effects.

    Parameters
    ----------
    filenameID : int
        Index of this file in the sorted file list.
    filename : str
        Basename of the .czi file (no path).
    cfg : PipelineConfig
        Fully-initialised pipeline configuration.
    norm_stats : dict, optional
        Pre-computed ``{channel: (p1, p99)}`` dict from
        :func:`patch_prep.compute_dataset_norm_stats`.
        Required when ``cfg.norm_mode == 'dataset'``, ignored otherwise.

    Returns
    -------
    list[pd.Series]
        One ``pd.Series`` per accepted patch (may be empty).
    """
    log.info("  Processing file %d/%d: %s", filenameID, cfg.end_ind - 1, filename)

    # ---- load + (rolling ball) + normalise + pad ----
    train_img, train_seg = patch_prep.load_and_pad(
        str(cfg.image_folder),
        str(cfg.cell_mask_folder) if cfg.cell_mask_folder is not None else None,
        filename,
        cfg.major_ch,
        pad_size=cfg.pad_size,
        norm_mode=cfg.norm_mode,
        norm_stats=norm_stats,
        norm_lo=cfg.norm_lo,
        norm_hi=cfg.norm_hi,
        file_type=cfg.file_type,
        seg_ch=cfg.seg_ch,
        seg_threshold=cfg.seg_threshold,
        seg_close_size=cfg.seg_close_size,
        seg_min_size_initial=cfg.seg_min_size_initial,
        seg_min_size_post_close=cfg.seg_min_size_post_close,
        seg_min_size_final=cfg.seg_min_size_final,
        rolling_ball_radius=cfg.rolling_ball_radius,
        norm_cell_scale=cfg.norm_cell_scale,
    )

    # ---- equivalent diameter from cell area (used to scale distance features) ----
    cell_area = float(np.sum(train_seg > 0))
    equiv_diam = 2.0 * np.sqrt(cell_area / np.pi) if cell_area > 0 else 1.0

    # ---- debug figure (accumulation plot) ----
    fig_accu, ax_accu = patch_prep.init_debug_fig(train_img, train_seg, dpi=cfg.dpi)

    # ---- grid ----
    x_num, y_num, x_0, y_0 = patch_prep.compute_grid(train_img.shape, cfg.patch_size)
    log.debug("    Grid: %d x %d patches (origin %d, %d)", x_num, y_num, x_0, y_0)

    rows: list[pd.Series] = []
    n_skipped_bounds = 0
    n_skipped_mask = 0

    for x_i, y_i, x_c, y_c in patch_prep.iter_grid_centers(
        x_num, y_num, x_0, y_0, cfg.patch_size
    ):
        if cfg.debug_flag:
            log.debug("    debug_flag=True – stopping after first grid position.")
            break

        # ---- big patch ----
        out = patch_prep.extract_big_patch(
            train_img, train_seg, x_c, y_c, cfg.double_ps
        )
        if out is None:
            n_skipped_bounds += 1
            continue
        patch_img, patch_seg, x_left, y_left = out

        # mask filter – level 1 (big patch)
        if patch_seg.mean() < cfg.mask_ratio / 64:
            n_skipped_mask += 1
            continue

        # ---- optional translation ----
        rand_tx, rand_ty = patch_prep.apply_optional_translation(
            cfg.rand_trans_flag, max_shift_px=cfg.max_shift_px
        )

        # ---- first crop ----
        big_crop_img, big_crop_seg, (cx_left_1, cy_up_1), _, _ = (
            patch_prep.first_crop_from_big(
                patch_img, patch_seg, cfg.patch_size, cfg.double_ps, rand_tx, rand_ty
            )
        )

        # mask filter – level 2 (big crop)
        if big_crop_seg.mean() < cfg.mask_ratio / 32:
            n_skipped_mask += 1
            continue

        # ---- optional rotation ----
        rot_img, rot_seg, rand_angle = patch_prep.apply_optional_rotation(
            big_crop_img, big_crop_seg,
            cfg.rand_rota_flag,
            max_angle_deg=cfg.max_angle_deg,
        )

        # ---- centre crop (final patch) ----
        crop_patch_img, crop_patch_seg, (cx_left_2, cy_up_2) = patch_prep.center_crop(
            rot_img, rot_seg, cfg.patch_size, cfg.half_ps
        )

        # mask filter – level 3 (final patch)
        if crop_patch_seg.mean() < cfg.mask_ratio:
            n_skipped_mask += 1
            continue

        # ---- save patch ----
        _prefix = f"{cfg.patch_prefix}_" if cfg.patch_prefix else ""
        crop_img_filename = (
            f"{_prefix}"
            f"f{str(filenameID).zfill(4)}"
            f"x{str(x_c).zfill(4)}"
            f"y{str(y_c).zfill(4)}"
            f"ps{cfg.patch_size}.tif"
        )
        patch_prep.save_patch(
            str(cfg.movie_partitioned_data_dir), crop_img_filename, crop_patch_img
        )

        # ---- polygon in full-image coords ----
        X_full, Y_full = patch_prep.compute_final_polygon_in_full_image(
            cfg.patch_size, rand_angle,
            cx_left_2, cy_up_2,
            x_left, y_left,
            cx_left_1, cy_up_1,
        )

        # overlay on accumulation plot
        plot_filename = (
            f"plot_grid_t{str(filenameID).zfill(4)}_xc{x_c}_yc{y_c}.png"
        )
        ax_accu[0].plot(X_full, Y_full, color="green")
        ax_accu[1].plot(X_full, Y_full, color="green")

        # ---- build record row ----
        s = patch_prep.make_record_row(
            str(cfg.image_folder), filename, filenameID, x_c, y_c,
            rand_angle, rand_tx, rand_ty,
            X_full, Y_full,
            str(cfg.movie_partitioned_data_dir), crop_img_filename,
            str(cfg.movie_plot_dir), plot_filename,
        )

        # ---- distance-to-boundary features (rotation- and scale-invariant) ----
        # train_seg is already padded; (y_c, x_c) are in padded coordinates.
        dist_feats = patch_prep.distance_to_boundary_features(
            train_seg, y_c, x_c, n_orientations=cfg.n_dist_orientations
        )
        dist_feats_norm = dist_feats / equiv_diam
        s["equiv_diam"] = equiv_diam
        for k, val in enumerate(dist_feats_norm):
            s[f"d{k:02d}"] = float(val)

        rows.append(s)

    log.info(
        "    → %d patches saved | %d skipped (bounds) | %d skipped (mask)",
        len(rows), n_skipped_bounds, n_skipped_mask,
    )

    # ---- save accumulation plot ----
    plot_path = cfg.movie_plot_dir / f"grid_t{str(filenameID).zfill(4)}.png"
    fig_accu.savefig(str(plot_path))
    plt.close(fig_accu)
    log.debug("    Accumulation plot saved → %s", plot_path)

    return rows


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

_BASE_RECORD_COLS = [
    "image_folder", "filename", "filenameID", "x_c", "y_c",
    "rand_angle", "rand_tx", "rand_ty",
    "x_corner1", "x_corner2", "x_corner3", "x_corner4",
    "y_corner1", "y_corner2", "y_corner3", "y_corner4",
    "movie_partitioned_data_dir", "crop_img_filename",
    "movie_plot_dir", "plot_filename",
    "equiv_diam",
]


def _record_cols(n_dist: int) -> list:
    """Return full column list including equiv_diam and d00…d{n_dist-1}."""
    return _BASE_RECORD_COLS + [f"d{k:02d}" for k in range(n_dist)]


def run_pipeline(cfg: PipelineConfig) -> pd.DataFrame:
    """Run the full patch-preparation pipeline.

    Iterates over .czi files in ``cfg.image_folder`` from index
    ``cfg.start_ind`` to ``cfg.end_ind`` (exclusive), extracts patches on a
    regular grid, applies optional augmentation, filters by mask coverage, and
    writes results to disk.

    A CSV record is written after **each file** so partial results are not lost
    if the run is interrupted.

    Parameters
    ----------
    cfg : PipelineConfig
        Fully-initialised configuration object.

    Returns
    -------
    pd.DataFrame
        Combined record table for all processed files and accepted patches.
    """
    log.info("=" * 60)
    log.info("Patch Prep Pipeline")
    log.info("  image_folder          : %s", cfg.image_folder)
    log.info("  cell_mask_folder      : %s", cfg.cell_mask_folder)
    log.info("  patch output dir      : %s", cfg.movie_partitioned_data_dir)
    log.info("  plot/csv output dir   : %s", cfg.movie_plot_dir)
    log.info("  condition             : %s", cfg.condition)
    log.info("  major_ch=%d  patch_size=%d  mask_ratio=%.2f",
             cfg.major_ch, cfg.patch_size, cfg.mask_ratio)
    log.info("  files [%d, %d)", cfg.start_ind, cfg.end_ind)
    log.info("  rand_trans=%s  rand_rota=%s  debug=%s",
             cfg.rand_trans_flag, cfg.rand_rota_flag, cfg.debug_flag)
    log.info("  norm_mode=%s  norm_lo=%.1f  norm_hi=%.1f",
             cfg.norm_mode, cfg.norm_lo, cfg.norm_hi)
    log.info("  rolling_ball_radius=%s", cfg.rolling_ball_radius)
    log.info("  norm_dataset_folder=%s", cfg.norm_dataset_folder or "(self)")
    log.info("  file_type=%s", cfg.file_type)
    log.info("  seg_ch=%s  seg_threshold=%.2f  seg_close_size=%d",
             cfg.seg_ch, cfg.seg_threshold, cfg.seg_close_size)
    log.info("=" * 60)

    filenames = patch_prep.list_image_files(str(cfg.image_folder), file_type=cfg.file_type)
    cols = _record_cols(cfg.n_dist_orientations)
    if not filenames:
        log.warning("No .czi files found in %s", cfg.image_folder)
        return pd.DataFrame(columns=cols)

    effective_end = min(cfg.end_ind, len(filenames))
    log.info("Found %d .czi file(s); processing indices %d–%d.",
             len(filenames), cfg.start_ind, effective_end - 1)

    # ---- dataset-level norm stats (computed once across ALL files in the run) ----
    norm_stats: Optional[Dict] = None
    if cfg.norm_mode == "dataset":
        if cfg.norm_dataset_folder is not None:
            # Use an external reference dataset (e.g. control) for percentile stats
            norm_folder = str(cfg.norm_dataset_folder)
            norm_files = patch_prep.list_image_files(norm_folder, file_type=cfg.file_type)
            log.info(
                "norm_mode='dataset': computing %d%%–%d%% percentile stats from "
                "reference folder '%s' (%d file(s)), channels %s …",
                int(cfg.norm_lo), int(cfg.norm_hi),
                norm_folder, len(norm_files), cfg.norm_channels,
            )
        else:
            norm_folder = str(cfg.image_folder)
            norm_files = filenames[cfg.start_ind:effective_end]
            log.info(
                "norm_mode='dataset': computing %d%%–%d%% percentile stats over "
                "%d file(s), channels %s …",
                int(cfg.norm_lo), int(cfg.norm_hi),
                len(norm_files), cfg.norm_channels,
            )
        norm_stats = patch_prep.compute_dataset_norm_stats(
            norm_folder,
            norm_files,
            channels=cfg.norm_channels,
            lo=cfg.norm_lo,
            hi=cfg.norm_hi,
            file_type=cfg.file_type,
        )
        for ch, (p1, p99) in norm_stats.items():
            log.info("    ch %d : p%.0f=%.6f  p%.0f=%.6f",
                     ch, cfg.norm_lo, p1, cfg.norm_hi, p99)

        # save stats to CSV for reproducibility
        stats_df = pd.DataFrame(
            [(ch, p1, p99) for ch, (p1, p99) in norm_stats.items()],
            columns=["channel", f"p{cfg.norm_lo:.0f}", f"p{cfg.norm_hi:.0f}"],
        )
        stats_csv = cfg.movie_plot_dir / "dataset_norm_stats.csv"
        stats_df.to_csv(str(stats_csv), index=False)
        log.info("  Dataset norm stats saved → %s", stats_csv)

    all_rows: list = []

    for filenameID in range(cfg.start_ind, effective_end):
        filename = filenames[filenameID]

        file_rows = _process_file(filenameID, filename, cfg, norm_stats=norm_stats)
        all_rows.extend(file_rows)

        # ---- write incremental CSV after each file ----
        record_so_far = pd.DataFrame(all_rows, columns=cols)
        csv_path = cfg.movie_plot_dir / (
            f"data_prep_record_{cfg.condition}_ch{cfg.major_ch}"
            f"_f_{cfg.start_ind}_to_{filenameID}.csv"
        )
        record_so_far.to_csv(str(csv_path), index=False)
        log.info("  CSV updated → %s", csv_path)

    full_record = pd.DataFrame(all_rows, columns=cols)
    log.info("Pipeline complete. Total patches: %d", len(full_record))
    return full_record
