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
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.pipeline.classification_pipeline import (
    ClassificationConfig,
    run_classification_pipeline,
)


# ---------------------------------------------------------------------------
# YAML → ClassificationConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path) -> ClassificationConfig:
    """Parse a YAML config file and return a :class:`ClassificationConfig`."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- input / output ----
    latents_csv = Path(str(_get("input",  "latents_csv", "")))
    out_dir     = Path(str(_get("output", "out_dir",     "results/classification")))

    # ---- labels ----
    label_col      = str(_get("labels", "label_col",      "annotation_label_name"))
    label_csv      = str(_get("labels", "label_csv",      "") or "")
    filename_col   = str(_get("labels", "filename_col",   "filename"))
    label_order    = _get("labels", "label_order",    None)
    exclude_labels = _get("labels", "exclude_labels", None)

    # ---- features ----
    feature_cols           = _get("features", "feature_cols",           None)
    include_mean_intensity = bool(_get("features", "include_mean_intensity", False))

    # ---- split ----
    split_strategy = str(_get("split", "strategy",     "from_csv"))
    test_size      = float(_get("split", "test_size",  0.2))
    random_state   = int(_get("split",  "random_state", 42))

    # ---- lightgbm ----
    n_estimators      = int(_get("lightgbm",   "n_estimators",      500))
    learning_rate     = float(_get("lightgbm", "learning_rate",     0.05))
    num_leaves        = int(_get("lightgbm",   "num_leaves",        31))
    min_child_samples = int(_get("lightgbm",   "min_child_samples", 20))
    class_weight      = str(_get("lightgbm",   "class_weight",      "balanced") or "balanced")
    n_cv_folds        = int(_get("lightgbm",   "n_cv_folds",        0))

    return ClassificationConfig(
        latents_csv=latents_csv,
        out_dir=out_dir,
        label_col=label_col,
        label_csv=label_csv,
        filename_col=filename_col,
        label_order=label_order,
        exclude_labels=exclude_labels,
        feature_cols=feature_cols,
        include_mean_intensity=include_mean_intensity,
        split_strategy=split_strategy,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        class_weight=class_weight,
        n_cv_folds=n_cv_folds,
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

    cfg = load_config(args.config)

    if args.dry_run:
        print("\n=== DRY RUN – resolved ClassificationConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo classification performed. Remove --dry_run to run for real.")
        return

    run_classification_pipeline(cfg)
    log.info("Done.")


if __name__ == "__main__":
    main()
