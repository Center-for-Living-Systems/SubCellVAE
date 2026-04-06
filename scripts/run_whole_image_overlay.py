"""
run_whole_image_overlay.py
==========================
Read a predictions_all.csv (output of run_cls_apply_from_config.py) and produce
per-source-image overlay PNGs: raw patch grid + coloured bounding boxes, one box
per patch coloured by the predicted label.

Coordinates are parsed directly from the patch filename:
    f{NNNN}x{xxxx}y{yyyy}ps{pp}.tif
The (x, y) values are the padded corner coordinates; pad_size is subtracted
to map back to the original (unpadded) image canvas.

Usage
-----
    python scripts/run_whole_image_overlay.py config/newdata_config/overlay_baseline_fa_lat8.yaml
    python scripts/run_whole_image_overlay.py config/... --root_folder /path/to/root
"""

from __future__ import annotations

import argparse
import logging
import re
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tiff

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

from subcellae.utils.config_utils import resolve_root


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default colour palettes (label_order index → hex colour)
# ---------------------------------------------------------------------------

_FA_COLORS = [
    "#e6194b",   # 0 Nascent Adhesion  – red
    "#f58231",   # 1 focal complex     – orange
    "#3cb44b",   # 2 focal adhesion    – green
    "#4363d8",   # 3 fibrillar adhesion – blue
    "#aaaaaa",   # 4 No adhesion       – grey
]

_POS_COLORS = [
    "#e6194b",   # 0 Cell Protruding Edge – red
    "#f58231",   # 1 Cell Periphery/other – orange
    "#3cb44b",   # 2 Lamella              – green
    "#4363d8",   # 3 Cell Body            – blue
]

_FALLBACK_COLORS = matplotlib.colormaps["tab10"].colors   # up to 10 classes


# ---------------------------------------------------------------------------
# Coordinate parsing
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r'^(f\d+)x(\d+)y(\d+)ps(\d+)\.tif$', re.IGNORECASE)


def _parse_coords(filename: str):
    """Return (img_id, x, y, ps) parsed from patch filename, or None."""
    m = _FNAME_RE.match(filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return None


# ---------------------------------------------------------------------------
# Core overlay logic
# ---------------------------------------------------------------------------

def _build_overlay(
    predictions_csv: str | Path,
    out_dir: str | Path,
    pad_size: int,
    label_order: list[str],
    colors: list[str],
    image_size: int,
    linewidth: float,
    dpi: int,
    title_prefix: str,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_csv)
    log.info("  Loaded %d rows from %s", len(df), predictions_csv)

    if "pred_label" not in df.columns:
        raise ValueError("'pred_label' column not found – run classifier apply first.")

    # ---- Parse coordinates ------------------------------------------------
    parsed = df["filename"].apply(_parse_coords)
    bad = parsed.isna().sum()
    if bad:
        log.warning("  %d filenames could not be parsed – skipped", bad)
    df = df[parsed.notna()].copy()
    coords = pd.DataFrame(parsed[parsed.notna()].tolist(),
                          columns=["img_id", "x", "y", "ps"],
                          index=df.index)
    df = pd.concat([df, coords], axis=1)

    # ---- Colour map -------------------------------------------------------
    # Build label_to_color keyed by string name (pred_label is stored as string)
    if label_order:
        n_classes = len(label_order)
        if len(colors) < n_classes:
            colors = list(colors) + [matplotlib.colors.to_hex(c)
                                      for c in _FALLBACK_COLORS[len(colors):]]
        label_to_color = {lbl: colors[i] for i, lbl in enumerate(label_order)}
        # Also accept integer index as fallback
        for i in range(n_classes):
            label_to_color[i] = colors[i]
    else:
        unique_labels = sorted(df["pred_label"].unique(), key=str)
        n_classes = len(unique_labels)
        if len(colors) < n_classes:
            colors = list(colors) + [matplotlib.colors.to_hex(c)
                                      for c in _FALLBACK_COLORS[len(colors):]]
        label_to_color = {lbl: colors[i] for i, lbl in enumerate(unique_labels)}

    # ---- Group by condition × image ---------------------------------------
    groups = df.groupby(["condition_name", "img_id"])
    log.info("  Generating overlays for %d images …", len(groups))

    for (cond, img_id), sub in groups:
        canvas = np.zeros((image_size, image_size), dtype=np.float32)

        # --- assemble patch grid ---
        for _, row in sub.iterrows():
            fpath = row["filepath"]
            if not os.path.exists(fpath):
                continue
            patch = tiff.imread(fpath).astype(np.float32)
            ps = int(row["ps"])
            x0 = int(row["x"]) - pad_size
            y0 = int(row["y"]) - pad_size
            x1 = x0 + ps
            y1 = y0 + ps
            # Clip to canvas bounds
            x0c, x1c = max(0, x0), min(image_size, x1)
            y0c, y1c = max(0, y0), min(image_size, y1)
            if x0c >= x1c or y0c >= y1c:
                continue
            px0, py0 = x0c - x0, y0c - y0
            patch_h, patch_w = patch.shape[-2], patch.shape[-1]
            if patch.ndim == 2:
                canvas[y0c:y1c, x0c:x1c] = patch[py0:py0+(y1c-y0c), px0:px0+(x1c-x0c)]
            else:
                canvas[y0c:y1c, x0c:x1c] = patch[0, py0:py0+(y1c-y0c), px0:px0+(x1c-x0c)]

        # Normalise canvas to [0, 1] for display
        cmax = canvas.max()
        if cmax > 0:
            canvas /= cmax

        # --- figure with coloured boxes ---
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{title_prefix} | {cond} | {img_id}", fontsize=8)
        ax.axis("off")

        for _, row in sub.iterrows():
            pred = row["pred_label"]
            color = label_to_color.get(pred, label_to_color.get(str(pred), "#ffffff"))
            ps = int(row["ps"])
            x0 = int(row["x"]) - pad_size
            y0 = int(row["y"]) - pad_size
            rect = mpatches.Rectangle(
                (x0, y0), ps, ps,
                linewidth=linewidth, edgecolor=color, facecolor="none", alpha=0.8,
            )
            ax.add_patch(rect)

        # --- legend ---
        legend_handles = []
        if label_order:
            for lbl in label_order:
                legend_handles.append(
                    mpatches.Patch(facecolor=label_to_color.get(lbl, "#ffffff"), label=lbl)
                )
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=5, loc="upper right",
                      framealpha=0.7, handlelength=1.2)

        fig.tight_layout(pad=0.5)
        out_name = f"overlay_{cond}_{img_id}.png"
        fig.savefig(out_dir / out_name, dpi=dpi)
        plt.close(fig)
        log.info("    saved %s", out_name)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_and_run(yaml_path: str | Path, root_folder: str | None = None):
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh)
    raw = resolve_root(raw, root_folder)

    def _get(section, key, default=None):
        return raw.get(section, {}).get(key, default)

    predictions_csv = str(_get("input", "predictions_csv", ""))
    out_dir         = str(_get("output", "out_dir", "results/overlay"))
    pad_size        = int(_get("misc", "pad_size",    64))
    image_size      = int(_get("misc", "image_size",  1024))
    linewidth       = float(_get("misc", "linewidth", 0.6))
    dpi             = int(_get("misc", "dpi",         300))
    title_prefix    = str(_get("misc", "title_prefix", "pred"))
    label_order     = _get("labels", "label_order", [])
    colors          = _get("labels", "colors", [])

    # Choose default colours based on number of labels
    if not colors:
        n = len(label_order)
        if n <= 4:
            colors = _POS_COLORS[:n]
        elif n <= 5:
            colors = _FA_COLORS[:n]
        else:
            colors = [matplotlib.colors.to_hex(c) for c in _FALLBACK_COLORS[:n]]

    log.info("Overlay config:")
    log.info("  predictions_csv : %s", predictions_csv)
    log.info("  out_dir         : %s", out_dir)
    log.info("  label_order     : %s", label_order)

    _build_overlay(
        predictions_csv=predictions_csv,
        out_dir=out_dir,
        pad_size=pad_size,
        label_order=label_order,
        colors=colors,
        image_size=image_size,
        linewidth=linewidth,
        dpi=dpi,
        title_prefix=title_prefix,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Whole-image overlay from predictions CSV.")
    p.add_argument("config", help="Path to YAML config file.")
    p.add_argument("--root_folder", default=None,
                   help="Override root_folder for all paths.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
    )

    load_and_run(args.config, root_folder=args.root_folder)
    log.info("Done.")


if __name__ == "__main__":
    main()
