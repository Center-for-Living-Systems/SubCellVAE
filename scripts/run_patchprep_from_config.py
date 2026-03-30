"""
run_patchprep_from_config.py
=============================
Load a YAML configuration file and run the patch-prep pipeline.

Usage
-----
    python scripts/run_patchprep_from_config.py config/config_npy_shared_no_norm.yaml
    python scripts/run_patchprep_from_config.py config/config_npy_shared_no_norm.yaml --dry_run
    python scripts/run_patchprep_from_config.py config/config_npy_shared_no_norm.yaml --log_level DEBUG

The YAML file drives every setting; no other arguments are required.
``--dry_run`` and ``--log_level`` are the only CLI flags accepted here —
everything else lives in the YAML.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys

from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root on sys.path so utils.patch_prep and patchprep_pipeline import cleanly
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.pipeline.patchprep_pipeline import PipelineConfig, run_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → PipelineConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> PipelineConfig:
    """Parse a YAML config file and return a :class:`PipelineConfig`.

    Parameters
    ----------
    yaml_path:
        Path to the ``.yaml`` / ``.yml`` config file.

    Returns
    -------
    PipelineConfig
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    # ---- convenience alias so we can write raw[section][key] safely ----
    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- paths ----
    image_folder      = Path(_get("paths", "image_folder"))
    raw_cell_mask     = _get("paths", "cell_mask_folder")   # may be None
    cell_mask_folder  = Path(raw_cell_mask) if raw_cell_mask is not None else None

    patch_output_dir = Path(_get("paths", "patch_output_dir"))
    plot_output_dir  = Path(_get("paths", "plot_output_dir"))

    # ---- build and return ----
    return PipelineConfig(
        # paths
        image_folder              = image_folder,
        cell_mask_folder          = cell_mask_folder,
        movie_partitioned_data_dir= patch_output_dir,
        movie_plot_dir            = plot_output_dir,

        # experiment
        condition   = _get("experiment", "condition", "shared"),
        major_ch    = int(_get("experiment", "major_ch", 1)),

        # input
        file_type   = _get("input", "file_type", "czi"),
        start_ind   = int(_get("input", "start_ind", 0)),
        end_ind     = int(_get("input", "end_ind", 5)),

        # patch geometry
        patch_size   = int(_get("patch",  "patch_size", 32)),
        mask_ratio   = float(_get("patch", "mask_ratio", 0.4)),
        pad_size     = int(_get("patch",  "pad_size", 64)),
        patch_prefix = str(_get("patch",  "patch_prefix", "") or ""),

        # normalization
        norm_mode     = _get("normalization", "norm_mode",     None),
        norm_channels = _get("normalization", "norm_channels", None),
        norm_lo       = float(_get("normalization", "norm_lo", 1.0)),
        norm_hi       = float(_get("normalization", "norm_hi", 99.0)),

        # segmentation
        seg_ch               = _get("segmentation", "seg_ch",               None),
        seg_threshold        = float(_get("segmentation", "seg_threshold",        0.2)),
        seg_close_size       = int(_get("segmentation",   "seg_close_size",       5)),
        seg_min_size_initial = int(_get("segmentation",   "seg_min_size_initial", 3)),
        seg_min_size_post_close = int(_get("segmentation","seg_min_size_post_close", 10)),
        seg_min_size_final   = int(_get("segmentation",   "seg_min_size_final",   30000)),

        # augmentation
        rand_trans_flag = bool(_get("augmentation", "rand_trans",    False)),
        max_shift_px    = int(_get("augmentation",  "max_shift_px",  0)),
        rand_rota_flag  = bool(_get("augmentation", "rand_rota",     False)),
        max_angle_deg   = float(_get("augmentation","max_angle_deg", 0.0)),

        # misc
        dpi         = int(_get("misc", "dpi",   256)),
        debug_flag  = bool(_get("misc", "debug", False)),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the patch-prep pipeline from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "config",
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved PipelineConfig and exit without processing.",
    )
    p.add_argument(
        "--log_level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. Overrides the value in the YAML file if given.",
    )
    p.add_argument(
        "--root_folder", default=None,
        help="Override root_folder for all paths. Useful when running on a different computer.",
    )
    return p.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # log level: CLI flag wins; fall back to YAML value; default INFO
    import yaml as _yaml
    with open(args.config, "r", encoding="utf-8") as _fh:
        _raw = _yaml.safe_load(_fh)
    yaml_log_level = _raw.get("misc", {}).get("log_level", "INFO")
    effective_log_level = args.log_level or yaml_log_level
    _setup_logging(effective_log_level)

    log = logging.getLogger(__name__)
    log.info("Loading config from: %s", args.config)

    cfg = load_config(args.config, root_folder=args.root_folder)

    if args.dry_run:
        print("\n=== DRY RUN – resolved PipelineConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<35} {v}")
        print("\nNo files processed. Remove --dry_run to run for real.")
        return

    record = run_pipeline(cfg)
    log.info(
        "Done. %d total patches accepted across files %d–%d.",
        len(record), cfg.start_ind, cfg.end_ind - 1,
    )

    # Copy config to the plot output directory for reproducibility
    cfg.movie_plot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.movie_plot_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.movie_plot_dir)


if __name__ == "__main__":
    main()
