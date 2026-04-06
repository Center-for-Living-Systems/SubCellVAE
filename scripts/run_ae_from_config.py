"""
run_ae_from_config.py
=====================
Load a YAML configuration file and run the autoencoder training pipeline.

Usage
-----
    python scripts/run_ae_from_config.py config/config_ae.yaml
    python scripts/run_ae_from_config.py config/config_ae.yaml --dry_run
    python scripts/run_ae_from_config.py config/config_ae.yaml --log_level DEBUG

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

from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline
from subcellae.utils.config_utils import resolve_root


# ---------------------------------------------------------------------------
# YAML → AEConfig
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, root_folder: str | None = None) -> AEConfig:
    """Parse a YAML config file and return an :class:`AEConfig`.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the ``.yaml`` / ``.yml`` config file.

    Returns
    -------
    AEConfig
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section: str, key: str, default=None):
        return raw.get(section, {}).get(key, default)

    # ---- data ----
    raw_patch_dirs = _get("data", "patch_dirs", [])
    patch_dirs = []
    for entry in raw_patch_dirs:
        if "channel_dirs" in entry:
            # Multi-channel mode: list of per-channel directories
            patch_dirs.append({
                "channel_dirs":  [str(d) for d in entry["channel_dirs"]],
                "condition":     int(entry.get("condition", 0)),
                "condition_name": str(entry.get("condition_name", "")),
            })
        else:
            # Single-channel mode (original behaviour)
            patch_dirs.append({
                "path":           str(entry["path"]),
                "condition":      int(entry.get("condition", entry.get("label", 0))),
                "condition_name": str(entry.get("condition_name", "")),
            })

    # ---- output ----
    result_dir = Path(_get("output", "result_dir", "results/ae"))

    # ---- model ----
    model_type    = str(_get("model", "model_type",    "ae"))
    latent_dim    = int(_get("model", "latent_dim",    8))
    input_ps      = int(_get("model", "input_ps",      32))
    no_ch         = int(_get("model", "no_ch",         1))
    BN_flag       = bool(_get("model", "BN_flag",      False))
    dropout_flag  = bool(_get("model", "dropout_flag", False))

    # VAE-specific
    out_activation = str(_get("model", "out_activation", "sigmoid"))
    beta           = float(_get("model", "beta",          1.0))
    beta_anneal    = bool(_get("model",  "beta_anneal",   False))
    recon_type     = str(_get("model",  "recon_type",     "mse"))

    # SemiSup-specific
    num_classes  = int(_get("model",   "num_classes",  6))
    lambda_recon = float(_get("model", "lambda_recon", 1.0))
    lambda_cls   = float(_get("model", "lambda_cls",   1.0))

    # Annotation (SemiSup per-patch labels)
    annotation_file = str(_get("annotation", "annotation_file", "") or "")
    label_col       = str(_get("annotation", "label_col",       "Classification"))
    filename_col    = str(_get("annotation", "filename_col",    "crop_img_filename"))
    label_order     = _get("annotation", "label_order", None)   # list or None

    # Second annotation (dual SemiSup)
    annotation_file_2 = str(_get("annotation", "annotation_file_2", "") or "")
    label_col_2       = str(_get("annotation", "label_col_2",       "Position"))
    filename_col_2    = str(_get("annotation", "filename_col_2",    "crop_img_filename"))
    label_order_2     = _get("annotation", "label_order_2", None)
    num_classes_2     = int(_get("model",      "num_classes_2",     0))
    lambda_cls_2      = float(_get("model",    "lambda_cls_2",      0.0))

    # Contrastive-specific
    proj_dim              = int(_get("model",   "proj_dim",              64))
    noise_prob            = float(_get("model", "noise_prob",            0.05))
    temperature           = float(_get("model", "temperature",           0.5))
    lambda_contrast       = float(_get("model", "lambda_contrast",       0.5))
    use_flip              = bool(_get("model",  "use_flip",              True))
    _isr                  = _get("model", "intensity_scale_range", [0.8, 1.2])
    intensity_scale_range = tuple(float(v) for v in _isr)

    # ---- training ----
    epochs         = int(_get("training",   "epochs",         200))
    lr             = float(_get("training", "lr",             1e-3))
    batch_size     = int(_get("training",   "batch_size",     128))
    val_split      = float(_get("training", "val_split",      0.2))
    loss_norm_flag = bool(_get("training",  "loss_norm_flag", False))
    group_split    = bool(_get("training",  "group_split",    True))
    weight_decay              = float(_get("training", "weight_decay",              1e-4))
    early_stopping_patience   = int(_get("training",   "early_stopping_patience",  0))
    min_epochs_for_best       = int(_get("training",   "min_epochs_for_best",       200))
    warmup_epochs             = int(_get("training",   "warmup_epochs",             200))

    # ---- reconstruction ----
    save_recon       = bool(_get("reconstruction", "save_recon",       True))
    recon_pad_size   = int(_get("reconstruction",  "recon_pad_size",   64))
    recon_image_size = int(_get("reconstruction",  "recon_image_size", 1024))

    # ---- misc ----
    device = str(_get("misc", "device", "auto"))

    return AEConfig(
        result_dir=result_dir,
        patch_dirs=patch_dirs,
        model_type=model_type,
        latent_dim=latent_dim,
        input_ps=input_ps,
        no_ch=no_ch,
        BN_flag=BN_flag,
        dropout_flag=dropout_flag,
        out_activation=out_activation,
        beta=beta,
        beta_anneal=beta_anneal,
        recon_type=recon_type,
        num_classes=num_classes,
        lambda_recon=lambda_recon,
        lambda_cls=lambda_cls,
        proj_dim=proj_dim,
        noise_prob=noise_prob,
        temperature=temperature,
        lambda_contrast=lambda_contrast,
        use_flip=use_flip,
        intensity_scale_range=intensity_scale_range,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_split=val_split,
        loss_norm_flag=loss_norm_flag,
        group_split=group_split,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        min_epochs_for_best=min_epochs_for_best,
        warmup_epochs=warmup_epochs,
        save_recon=save_recon,
        recon_pad_size=recon_pad_size,
        recon_image_size=recon_image_size,
        device=device,
        annotation_file=annotation_file,
        label_col=label_col,
        filename_col=filename_col,
        label_order=label_order,
        annotation_file_2=annotation_file_2,
        label_col_2=label_col_2,
        filename_col_2=filename_col_2,
        label_order_2=label_order_2,
        num_classes_2=num_classes_2,
        lambda_cls_2=lambda_cls_2,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the autoencoder training pipeline from a YAML config file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "config",
        help="Path to the YAML configuration file.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved AEConfig and exit without training.",
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
    with open(args.config, "r", encoding="utf-8") as fh:
        _raw = yaml.safe_load(fh)
    yaml_log_level = _raw.get("misc", {}).get("log_level", "INFO")
    effective_log_level = args.log_level or yaml_log_level
    _setup_logging(effective_log_level)

    log = logging.getLogger(__name__)
    log.info("Loading config from: %s", args.config)

    cfg = load_config(args.config, root_folder=args.root_folder)

    if args.dry_run:
        print("\n=== DRY RUN – resolved AEConfig ===")
        for k, v in vars(cfg).items():
            print(f"  {k:<30} {v}")
        print("\nNo training performed. Remove --dry_run to run for real.")
        return

    run_ae_pipeline(cfg)

    # Copy config to the result directory for reproducibility
    cfg.result_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, cfg.result_dir / Path(args.config).name)
    log.info("Config copied to: %s", cfg.result_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
