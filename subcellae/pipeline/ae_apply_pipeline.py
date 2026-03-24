"""
ae_apply_pipeline.py
====================
Inference-only autoencoder pipeline for applying a trained model to a new,
unlabelled dataset.

Loads a saved ``model_final.pt`` (or any ``model_*.pt`` checkpoint), runs all
patches through the encoder, and writes a ``latents_newdata.csv`` in the same
format as ``latents.csv`` produced by :func:`run_ae_pipeline` — but with
``split="newdata"`` for every row (no labels, no train/val split).

Optionally saves reconstruction images (patch tifs + whole-image canvases).

Typical use-case
----------------
You have a new imaging experiment (no annotations) and four trained AEs.
Run this pipeline once per AE to obtain latent CSVs, then feed those into
:func:`run_cls_apply_pipeline` to get predicted labels.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import ConcatDataset, DataLoader

from subcellae.modelling.dataset import PatchDataset
from subcellae.modelling.autoencoders import AE, VAE32, SemiSupAE, ContrastiveAE
from subcellae.pipeline.ae_pipeline import (
    _extract_group_key,
    _parse_patch_coords,
    _extract_latents,
)

log = logging.getLogger(__name__)

_CLASS_TO_TYPE = {
    "AE":           "ae",
    "VAE32":        "vae",
    "SemiSupAE":    "semisup",
    "ContrastiveAE": "contrastive",
}


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class AEApplyConfig:
    """All parameters for one AE-apply run.

    Parameters
    ----------
    model_pt : Path
        Path to the saved model checkpoint (``model_final.pt`` or similar).
        The model type is inferred automatically from the class name stored
        inside the checkpoint.
    patch_dirs : list[dict]
        Each entry must have ``path`` (str), ``condition`` (int), and
        optionally ``condition_name`` (str).  Labels are never loaded.
    out_dir : Path
        Directory where ``latents_newdata.csv`` (and optional recon outputs)
        are written.
    batch_size : int
    device : str
        ``"auto"`` selects CUDA if available, else CPU.
    save_recon : bool
        Write reconstruction tifs and comparison PNGs.
    recon_pad_size : int
        Padding subtracted when mapping patch centres to whole-image coords.
    recon_image_size : int
        Canvas size (pixels) for whole-image reconstruction tifs.
    """

    # --- required ---
    model_pt: Path
    patch_dirs: list        # [{path, condition, condition_name}, ...]
    out_dir: Path

    # --- inference ---
    batch_size: int = 128
    device: str = "auto"

    # --- reconstruction ---
    save_recon: bool = False
    recon_pad_size: int = 64
    recon_image_size: int = 1024

    def __post_init__(self):
        self.model_pt  = Path(self.model_pt)
        self.out_dir   = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(model_pt: Path, device: str) -> tuple:
    """Load model from checkpoint; return (model, model_type_str).

    The checkpoint is expected to be a full model object (saved with
    ``torch.save(model, path)``).  The model type is inferred from the
    class name.
    """
    log.info("Loading model from %s", model_pt)
    model = torch.load(str(model_pt), map_location=device)
    cls_name = type(model).__name__
    model_type = _CLASS_TO_TYPE.get(cls_name)
    if model_type is None:
        raise ValueError(
            f"Unrecognised model class {cls_name!r}. "
            f"Expected one of {list(_CLASS_TO_TYPE.keys())}."
        )
    log.info("  Detected model type: %s (%s)", model_type, cls_name)
    model.eval()
    model.to(device)
    return model, model_type


def _save_latent_csv_newdata(result: dict, datasets: list, out_dir: Path) -> Path:
    """Build and save the newdata latents CSV.

    Columns mirror ``latents.csv`` from ae_pipeline, but:
      - ``split`` is always ``"newdata"``
      - ``annotation_label`` and ``annotation_label_name`` are always ``-1`` / ``""``
    """
    cond_to_name = {ds.condition: ds.condition_name for ds in datasets}
    latents = result["latents"]
    latent_dim = latents.shape[1]

    rows = []
    for i, path in enumerate(result["paths"]):
        condition = result["conditions"][i]
        row = {
            "filename":              Path(path).name,
            "filepath":              path,
            "condition":             condition,
            "condition_name":        cond_to_name.get(condition, str(condition)),
            "group":                 _extract_group_key(path),
            "split":                 "newdata",
            "recon_mse":             result["recon_mse"][i],
            "mean_intensity":        result["mean_intensity"][i],
            "norm_mse":              result["norm_mse"][i],
            "annotation_label":      -1,
            "annotation_label_name": "",
        }
        for d, val in enumerate(latents[i]):
            row[f"z_{d}"] = float(val)
        rows.append(row)

    meta_cols   = ["filename", "filepath", "condition", "condition_name", "group", "split"]
    metric_cols = ["recon_mse", "mean_intensity", "norm_mse"]
    latent_cols = [f"z_{d}" for d in range(latent_dim)]
    ann_cols    = ["annotation_label", "annotation_label_name"]

    df = pd.DataFrame(rows, columns=meta_cols + metric_cols + latent_cols + ann_cols)
    out_path = out_dir / "latents_newdata.csv"
    df.to_csv(out_path, index=False)
    log.info("Latents CSV saved → %s  (%d patches)", out_path, len(df))
    return out_path


def _save_reconstructions_newdata(result: dict, out_dir: Path,
                                  recon_pad_size: int, recon_image_size: int) -> None:
    """Save reconstruction tifs and comparison PNGs for the new data."""
    recon_dir  = out_dir / "recon"
    patch_dir  = recon_dir / "patches"
    image_dir  = recon_dir / "images"
    visual_dir = recon_dir / "visual"
    for d in (patch_dir, image_dir, visual_dir):
        d.mkdir(parents=True, exist_ok=True)

    pad = recon_pad_size
    canvas_data: dict = defaultdict(list)

    for i, path in enumerate(result["paths"]):
        fname   = Path(path).name
        raw_p   = result["raws"][i]
        recon_p = result["recons"][i]

        tifffile.imwrite(str(patch_dir / f"raw_{fname}"),   raw_p)
        tifffile.imwrite(str(patch_dir / f"recon_{fname}"), recon_p)

        coords = _parse_patch_coords(fname)
        if coords is None:
            continue

        group, x_c, y_c, ps = coords
        half = ps // 2
        r0 = y_c - half - pad
        r1 = y_c + half - pad
        c0 = x_c - half - pad
        c1 = x_c + half - pad

        if r0 < 0 or c0 < 0:
            continue
        canvas_data[group].append((r0, r1, c0, c1, raw_p, recon_p))

    img_size = recon_image_size
    for group, entries in sorted(canvas_data.items()):
        raw_canvas   = np.zeros((img_size, img_size), dtype=np.float32)
        recon_canvas = np.zeros((img_size, img_size), dtype=np.float32)

        for r0, r1, c0, c1, raw_p, recon_p in entries:
            r1 = min(r1, img_size)
            c1 = min(c1, img_size)
            raw_canvas[r0:r1, c0:c1]   = raw_p[:r1-r0, :c1-c0]
            recon_canvas[r0:r1, c0:c1] = recon_p[:r1-r0, :c1-c0]

        tifffile.imwrite(str(image_dir / f"raw_{group}.tif"),   raw_canvas)
        tifffile.imwrite(str(image_dir / f"recon_{group}.tif"), recon_canvas)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(group, fontsize=13, fontweight="bold")
        axes[0].imshow(raw_canvas,   cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Raw")
        axes[0].axis("off")
        axes[1].imshow(recon_canvas, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(str(visual_dir / f"{group}_comparison.png"), dpi=150)
        plt.close(fig)

    log.info("Reconstruction output saved → %s  (%d source images)",
             recon_dir, len(canvas_data))


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def run_ae_apply_pipeline(cfg: AEApplyConfig) -> Path:
    """Apply a trained AE to a new unlabelled dataset.

    Returns
    -------
    Path
        Path to the saved ``latents_newdata.csv``.
    """
    log.info("=" * 60)
    log.info("AE Apply Pipeline (inference only)")
    log.info("  model_pt   : %s", cfg.model_pt)
    log.info("  out_dir    : %s", cfg.out_dir)
    log.info("  patch_dirs : %d director%s",
             len(cfg.patch_dirs), "y" if len(cfg.patch_dirs) == 1 else "ies")
    for entry in cfg.patch_dirs:
        log.info("    path=%-50s  condition=%s (%s)",
                 entry.get("path", "?"),
                 entry.get("condition", "?"),
                 entry.get("condition_name", ""))
    log.info("  device     : %s", cfg.device)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, model_type = _load_model(cfg.model_pt, cfg.device)

    # ------------------------------------------------------------------
    # 2. Build datasets (no annotation file — unlabelled)
    # ------------------------------------------------------------------
    def _channel_expand(x: np.ndarray) -> np.ndarray:
        return np.expand_dims(x, 0)

    datasets = []
    for entry in cfg.patch_dirs:
        path           = entry["path"]
        condition      = int(entry.get("condition", entry.get("label", 0)))
        condition_name = str(entry.get("condition_name", str(condition)))
        ds = PatchDataset(
            path,
            condition=condition,
            condition_name=condition_name,
            annotation_file=None,
            transform=_channel_expand,
        )
        log.info("  Loaded %d patches from %s  condition=%d (%s)",
                 len(ds), path, condition, condition_name)
        datasets.append(ds)

    if not datasets:
        raise ValueError("patch_dirs is empty; nothing to apply the model to.")

    full_dataset = ConcatDataset(datasets)
    log.info("  Total patches: %d", len(full_dataset))

    # ------------------------------------------------------------------
    # 3. DataLoader
    # ------------------------------------------------------------------
    loader = DataLoader(
        full_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    log.info("Running inference …")
    result = _extract_latents(model, loader, cfg.device, model_type)

    # ------------------------------------------------------------------
    # 5. Save latents CSV
    # ------------------------------------------------------------------
    csv_path = _save_latent_csv_newdata(result, datasets, cfg.out_dir)

    # ------------------------------------------------------------------
    # 6. Optionally save reconstructions
    # ------------------------------------------------------------------
    if cfg.save_recon:
        log.info("Saving reconstructions …")
        _save_reconstructions_newdata(
            result, cfg.out_dir, cfg.recon_pad_size, cfg.recon_image_size
        )

    log.info("AE Apply Pipeline complete.")
    return csv_path
