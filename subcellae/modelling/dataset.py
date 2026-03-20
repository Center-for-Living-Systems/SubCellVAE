"""
dataset.py
==========
PyTorch Dataset classes for .tif patch files.

The primary class is :class:`PatchDataset`, which always returns a 4-tuple::

    (image, condition, annotation_label, path)

where:

* ``image``            – (1, H, W) float32 tensor, values in [0, 1]
* ``condition``        – integer condition ID for the whole directory
                         (e.g. 0 = control, 1 = ycomp).  Used to distinguish
                         experimental groups; ignored by plain AE/VAE training.
* ``annotation_label`` – per-patch integer class from an optional annotation
                         file (e.g. FA type: 0–4), or ``-1`` if the patch was
                         not annotated.  Used by :func:`semisup_ae_loss`.
* ``path``             – absolute path to the .tif file (str).

:class:`TIFFDataset` is kept as a backward-compatible wrapper that returns the
old 3-tuple ``(image, condition, path)`` (``label`` is now called
``condition`` internally but the argument name is unchanged).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------------------------
# Filename normalisation helper
# ---------------------------------------------------------------------------
# Patch files produced by the pipeline use underscore before the coordinate
# block: control_f0001x0112y0496ps32.tif
# Annotation CSVs (unique_ID column) use a hyphen: control-f0001x0112y0496ps32.tif
# This regex converts the patch filename to the annotation-CSV style so that
# the per-patch label lookup succeeds.
_COORD_UNDERSCORE = re.compile(r'_(f\d+x\d+y\d+ps\d+\.tiff?)$', re.IGNORECASE)


def _patch_name_to_annotation_key(filename: str) -> str:
    """Convert a patch filename to the annotation-CSV key style.

    ``control_f0001x0112y0496ps32.tif`` → ``control-f0001x0112y0496ps32.tif``
    """
    return _COORD_UNDERSCORE.sub(r'-\1', Path(filename).name)


# ---------------------------------------------------------------------------
# Primary unified class
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """Unified dataset for .tif image patches.

    Always returns ``(image, condition, annotation_label, path)``.

    Parameters
    ----------
    root_dir : str
        Directory containing .tif patch files.
    condition : int
        Integer ID for the experimental condition of this directory
        (e.g. ``0`` = control, ``1`` = ycomp).  Applied uniformly to every
        patch in the directory.
    condition_name : str
        Human-readable name for the condition (used in log output only).
    annotation_file : str or None
        Path to a CSV or Excel file with per-patch annotation labels.
        If ``None`` (default), ``annotation_label`` is ``-1`` for every patch.
    label_col : str
        Column in the annotation file to use as the class label
        (e.g. ``"Classification"`` or ``"position"``).
    filename_col : str
        Column in the annotation file that holds patch basenames
        (e.g. ``"crop_img_filename"`` or ``"unique_ID"``).
    label_order : list[str] or None
        Ordered list of string labels that defines the integer mapping
        (index 0 → class 0, …).  If ``None``, unique values are sorted
        alphabetically.
    transform : callable or None
        Applied to the raw ``(H, W)`` float32 numpy array before it is
        wrapped in a tensor.
    """

    def __init__(
        self,
        root_dir: str,
        condition: int = 0,
        condition_name: str = "",
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        transform=None,
    ):
        self.root_dir       = root_dir
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.transform      = transform

        # ---- build annotation lookup (filename → int label) ----
        fname_to_annotation: dict[str, int] = {}
        self.label_order   = label_order or []
        self.label_to_int: dict[str, int] = {}
        self.num_classes   = 0

        if annotation_file:
            ann_path = Path(annotation_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))

            # normalise filename column to basename
            ann_df[filename_col] = (
                ann_df[filename_col].astype(str).apply(lambda p: Path(p).name)
            )
            fname_to_str = dict(
                zip(ann_df[filename_col], ann_df[label_col].astype(str))
            )

            if not self.label_order:
                self.label_order = sorted(
                    {v for v in fname_to_str.values() if v and v != "nan"}
                )
            self.label_to_int = {lbl: i for i, lbl in enumerate(self.label_order)}
            self.num_classes  = len(self.label_order)

            fname_to_annotation = {
                fname: self.label_to_int.get(str_lbl, -1)
                for fname, str_lbl in fname_to_str.items()
            }

        # ---- load images ----
        all_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(("tif", "tiff"))
        ])

        self.data              = []
        self.paths             = []
        self.annotation_labels = []

        for img_path in all_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                if self.transform:
                    image = self.transform(image)
                image = torch.tensor(image, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} – {e}")
                continue

            ann_label = fname_to_annotation.get(
                _patch_name_to_annotation_key(img_path), -1
            )

            self.data.append(image)
            self.paths.append(img_path)
            self.annotation_labels.append(ann_label)

        n_ann = sum(1 for l in self.annotation_labels if l >= 0)
        print(
            f"PatchDataset [{self.condition_name}]: {len(self.data)} patches, "
            f"condition={condition}, "
            f"{n_ann} annotated / {len(self.data) - n_ann} unlabelled"
            + (f" ({label_col})" if annotation_file else "")
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.condition,
            self.annotation_labels[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

class TIFFDataset(Dataset):
    """Backward-compatible dataset returning ``(image, condition, path)``.

    Wraps :class:`PatchDataset`.  The argument ``label`` is accepted for
    historical reasons and maps to ``condition``.
    """

    def __init__(self, root_dir, label: int = 0, transform=None):
        self._ds = PatchDataset(root_dir, condition=label, transform=transform)
        # expose paths so grouped-split helpers can access them
        self.paths = self._ds.paths

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        image, condition, _, path = self._ds[idx]
        return image, condition, path


class AnnotatedTIFFDataset(Dataset):
    """Backward-compatible annotated dataset returning ``(image, annotation_label, path)``.

    Wraps :class:`PatchDataset`.
    """

    def __init__(
        self,
        root_dir,
        annotation_file,
        label_col,
        filename_col="crop_img_filename",
        label_order=None,
        transform=None,
    ):
        self._ds = PatchDataset(
            root_dir,
            condition=0,
            annotation_file=annotation_file,
            label_col=label_col,
            filename_col=filename_col,
            label_order=label_order,
            transform=transform,
        )
        self.label_order  = self._ds.label_order
        self.label_to_int = self._ds.label_to_int
        self.num_classes  = self._ds.num_classes
        self.paths        = self._ds.paths

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        image, _, annotation_label, path = self._ds[idx]
        return image, annotation_label, path
