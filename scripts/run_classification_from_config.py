"""
run_classification_from_config.py
==================================
Load a YAML configuration file and run the LightGBM classification pipeline.

Usage
-----
    python scripts/run_classification_from_config.py config/config_classification.yaml
    python scripts/run_classification_from_config.py config/config_classification.yaml --dry_run
    python scripts/run_classification_from_config.py config/config_classification.yaml --log_level DEBUG
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

from subcellae.pipeline.classification_pipeline import (
    ClassificationConfig,
    run_classification_pipeline,
)
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → ClassificationConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> ClassificationConfig:
    """Parse a YAML config file and return a :class:`ClassificationConfig`."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- input / output ----
    latents_csv = Path(str(_get("input",  "latents_csv", "")))
    out_dir     = Path(str(_get("output", "out_dir",     "results/classification")))

    # ---- labels ----
    label_col      = str(_get("labels", "label_col",      "annotation_label_name"))
    label_csv      = str(_get("labels", "label_csv",      "") or "")
    filename_col   = str(_get("labels", "filename_col",   "filename"))
    label_order             = _get("labels", "label_order",             None)
    exclude_labels          = _get("labels", "exclude_labels",          None)
    metrics_exclude_labels  = _get("labels", "metrics_exclude_labels",  None)

    # ---- features ----
    feature_cols           = _get("features", "feature_cols",           None)
    feature_prefix         = str(_get("features", "feature_prefix",     "z_"))
    include_mean_intensity = bool(_get("features", "include_mean_intensity", False))

    # ---- split ----
    split_strategy = str(_get("split", "strategy",     "from_csv"))
    test_size      = float(_get("split", "test_size",  0.2))
    random_state   = int(_get("split",  "random_state", 42))

    # ---- classifier ----
    classifier_type   = str(_get("classifier", "type",              "lgbm"))

    # ---- lightgbm ----
    n_estimators      = int(_get("lightgbm",   "n_estimators",      500))
    learning_rate     = float(_get("lightgbm", "learning_rate",     0.05))
    num_leaves        = int(_get("lightgbm",   "num_leaves",        31))
    min_child_samples = int(_get("lightgbm",   "min_child_samples", 20))
    class_weight      = str(_get("lightgbm",   "class_weight",      "balanced") or "balanced")
    n_cv_folds        = int(_get("lightgbm",   "n_cv_folds",        0))

    # ---- svm ----
    svm_C     = float(_get("svm", "C",     10.0))
    svm_gamma = str(_get("svm",   "gamma", "scale"))

    # ---- mlp ----
    mlp_hidden_layers = _get("mlp", "hidden_layers", None)
    mlp_max_iter      = int(_get("mlp", "max_iter",  1000))

    # ---- knn ----
    knn_n_neighbors = int(_get("knn", "n_neighbors", 10))

    # ---- distance features ----
    dist_patch_prep_dirs  = _get("dist_features", "patch_prep_dirs",    None) or []
    dist_feature_weight   = float(_get("dist_features", "feature_weight", 100.0))

    # ---- patch sorting ----
    sort_labelled   = bool(_get("patch_sort", "sort_labelled",   True))
    sort_unlabelled = bool(_get("patch_sort", "sort_unlabelled", False))

    return ClassificationConfig(
        latents_csv=latents_csv,
        out_dir=out_dir,
        label_col=label_col,
        label_csv=label_csv,
        filename_col=filename_col,
        label_order=label_order,
        exclude_labels=exclude_labels,
        metrics_exclude_labels=metrics_exclude_labels,
        feature_cols=feature_cols,
        feature_prefix=feature_prefix,
        include_mean_intensity=include_mean_intensity,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        classifier_type=classifier_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        class_weight=class_weight,
        n_cv_folds=n_cv_folds,
        svm_C=svm_C,
        svm_gamma=svm_gamma,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_max_iter=mlp_max_iter,
        knn_n_neighbors=knn_n_neighbors,
        dist_patch_prep_dirs=dist_patch_prep_dirs,
        dist_feature_weight=dist_feature_weight,
        sort_labelled=sort_labelled,
        sort_unlabelled=sort_unlabelled,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the LightGBM classification pipeline from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to the YAML configuration file.")
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the resolved ClassificationConfig and exit without running.",
    )
    p.add_argument(
        "--log_level", default=None,
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

    with open(args.config, "r", encoding="utf-8") as fh:
        _raw = yaml.safe_load(fh)
    yaml_log_level      = _raw.get("misc", {}).get("log_level", "INFO")
    effective_log_level = args.log_level or yaml_log_level
    _setup_logging(effective_log_level)

    log = logging.getLogger(__name__)
    log.info("Loading config from: %s", args.config)

    cfg = load_config(args.config, root_folder=args.root_folder)

    if args.dry_run:
        print("\n=== DRY RUN – resolved ClassificationConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo classification performed. Remove --dry_run to run for real.")
        return

    run_classification_pipeline(cfg)

    # Copy config to the output directory for reproducibility
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
