"""
run_cls_apply_from_config.py
============================
Load a YAML config and apply a trained LightGBM classifier to a new (unlabelled)
latents CSV.

Usage
-----
    python scripts/run_cls_apply_from_config.py config/newdata/cls_apply_baseline_fa_lat8.yaml
    python scripts/run_cls_apply_from_config.py config/...  --dry_run
    python scripts/run_cls_apply_from_config.py config/...  --log_level DEBUG
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

from subcellae.pipeline.cls_apply_pipeline import ClsApplyConfig, run_cls_apply_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → ClsApplyConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> ClsApplyConfig:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- input / output ----
    latents_csv = Path(str(_get("input",  "latents_csv",  "")))
    model_pkl   = Path(str(_get("model",  "model_pkl",    "")))
    out_dir     = Path(str(_get("output", "out_dir",      "results/cls_apply")))

    # ---- optional model ----
    umap_model_pkl_raw = _get("model", "umap_model_pkl", None)
    umap_model_pkl     = Path(str(umap_model_pkl_raw)) if umap_model_pkl_raw else None

    # ---- labels ----
    label_order = _get("labels", "label_order", None)

    # ---- features ----
    feature_cols   = _get("features", "feature_cols", None)

    # ---- distance features ----
    dist_patch_prep_dirs = _get("dist_features", "patch_prep_dirs",    None) or []
    dist_feature_weight  = float(_get("dist_features", "feature_weight", 100.0))

    # ---- UMAP ----
    umap_n_neighbors   = int(_get("umap",   "n_neighbors",  15))
    umap_min_dist      = float(_get("umap", "min_dist",     0.1))
    umap_random_state  = int(_get("umap",   "random_state", 42))

    return ClsApplyConfig(
        latents_csv=latents_csv,
        model_pkl=model_pkl,
        out_dir=out_dir,
        label_order=label_order,
        umap_model_pkl=umap_model_pkl,
        feature_cols=feature_cols,
        dist_patch_prep_dirs=dist_patch_prep_dirs,
        dist_feature_weight=dist_feature_weight,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply a trained LightGBM classifier to new unlabelled latents.",
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
        print("\n=== DRY RUN – resolved ClsApplyConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo classification performed.  Remove --dry_run to run for real.")
        return

    run_cls_apply_pipeline(cfg)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
