"""
run_analysis_from_config.py
===========================
Load a YAML configuration file and run the post-training analysis pipeline.

Usage
-----
    python scripts/run_analysis_from_config.py config/config_analysis.yaml
    python scripts/run_analysis_from_config.py config/config_analysis.yaml --dry_run
    python scripts/run_analysis_from_config.py config/config_analysis.yaml --log_level DEBUG

The YAML file drives every setting; no other arguments are required.
``--dry_run`` and ``--log_level`` are the only CLI flags accepted here —
everything else lives in the YAML.
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

from subcellae.pipeline.analysis_pipeline import AnalysisConfig, run_analysis_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → AnalysisConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> AnalysisConfig:
    """Parse a YAML config file and return an :class:`AnalysisConfig`."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- input / output ----
    latents_csv  = Path(str(_get("input",  "latents_csv",  "")))
    out_dir      = Path(str(_get("output", "out_dir",      "results/analysis")))
    split_filter = str(_get("input", "split_filter", "all"))

    # ---- embedding ----
    umap_methods      = list(_get("embedding", "methods", ["UMAP"]))
    umap_n_neighbors  = int(_get("embedding",  "umap_n_neighbors",  15))
    umap_min_dist     = float(_get("embedding","umap_min_dist",     0.1))
    umap_random_state = int(_get("embedding",  "umap_random_state", 42))
    phate_k           = int(_get("embedding",  "phate_k",           5))

    # ---- clustering ----
    kmeans_enabled    = bool(_get("clustering", "kmeans_enabled",    True))
    kmeans_n_clusters = int(_get("clustering",  "kmeans_n_clusters", 5))
    dbscan_enabled    = bool(_get("clustering", "dbscan_enabled",    False))
    dbscan_eps        = float(_get("clustering","dbscan_eps",        0.5))
    dbscan_min_samples = int(_get("clustering", "dbscan_min_samples",10))
    boxplot_kind      = str(_get("clustering",  "boxplot_kind",      "box"))

    # ---- label orders ----
    label_orders = raw.get("label_orders", {}) or {}
    annotation_label_order = label_orders.get("annotation_label_name", None)
    condition_name_order   = label_orders.get("condition_name",         None)

    return AnalysisConfig(
        latents_csv=latents_csv,
        out_dir=out_dir,
        split_filter=split_filter,
        umap_methods=umap_methods,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_random_state=umap_random_state,
        phate_k=phate_k,
        kmeans_enabled=kmeans_enabled,
        kmeans_n_clusters=kmeans_n_clusters,
        dbscan_enabled=dbscan_enabled,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        boxplot_kind=boxplot_kind,
        annotation_label_order=annotation_label_order,
        condition_name_order=condition_name_order,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the analysis pipeline from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("config", help="Path to the YAML configuration file.")
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print the resolved AnalysisConfig and exit without running.",
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
        print("\n=== DRY RUN – resolved AnalysisConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo analysis performed. Remove --dry_run to run for real.")
        return

    run_analysis_pipeline(cfg)

    # Copy config to the output directory for reproducibility
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.out_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
