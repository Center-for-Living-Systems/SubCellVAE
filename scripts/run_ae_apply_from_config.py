"""
run_ae_apply_from_config.py
===========================
Load a YAML config and apply a trained AE to a new (unlabelled) dataset.

Usage
-----
    python scripts/run_ae_apply_from_config.py config/newdata/ae_apply_baseline.yaml
    python scripts/run_ae_apply_from_config.py config/... --dry_run
    python scripts/run_ae_apply_from_config.py config/... --log_level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.pipeline.ae_apply_pipeline import AEApplyConfig, run_ae_apply_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → AEApplyConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> AEApplyConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- model ----
    model_pt = Path(str(_get("model", "model_pt", "")))

    # ---- data ----
    patch_dirs = _get("data", "patch_dirs", [])
    patch_dirs = [
        {
            "path":           str(entry["path"]),
            "condition":      int(entry.get("condition", entry.get("label", 0))),
            "condition_name": str(entry.get("condition_name", "")),
        }
        for entry in patch_dirs
    ]

    # ---- output ----
    out_dir = Path(str(_get("output", "out_dir", "results/ae_apply")))

    # ---- inference ----
    batch_size = int(_get("inference", "batch_size", 128))
    device     = str(_get("misc", "device", "auto"))

    # ---- reconstruction ----
    save_recon       = bool(_get("reconstruction", "save_recon",       False))
    recon_pad_size   = int(_get("reconstruction",  "recon_pad_size",   64))
    recon_image_size = int(_get("reconstruction",  "recon_image_size", 1024))

    return AEApplyConfig(
        model_pt=model_pt,
        patch_dirs=patch_dirs,
        out_dir=out_dir,
        batch_size=batch_size,
        device=device,
        save_recon=save_recon,
        recon_pad_size=recon_pad_size,
        recon_image_size=recon_image_size,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a trained AE to new unlabelled data (inference only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to YAML config file.")
    p.add_argument("--dry_run",  action="store_true",
                   help="Print resolved config and exit without running.")
    p.add_argument("--log_level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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

    with open(args.config, "r", encoding="utf-8") as fh:
        _raw = yaml.safe_load(fh)
    yaml_log_level = _raw.get("misc", {}).get("log_level", "INFO")
    _setup_logging(args.log_level or yaml_log_level)

    log = logging.getLogger(__name__)
    log.info("Loading config from: %s", args.config)

    cfg = load_config(args.config, root_folder=args.root_folder)

    if args.dry_run:
        print("\n=== DRY RUN – resolved AEApplyConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo inference performed.  Remove --dry_run to run for real.")
        return

    run_ae_apply_pipeline(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
