"""
dataset.py
==========
PyTorch Dataset classes for .tif patch files.

The primary class is :class:`PatchDataset`, which always returns a 5-tuple::

    (image, condition, annotation_label, annotation_label_2, path)

where:

* ``image``              – (1, H, W) float32 tensor, values in [0, 1]
* ``condition``          – integer condition ID for the whole directory
                           (e.g. 0 = control, 1 = ycomp).  Used to distinguish
                           experimental groups; ignored by plain AE/VAE training.
* ``annotation_label``   – per-patch integer class from the primary annotation
                           file (e.g. FA type: 0–4), or ``-1`` if unlabelled.
                           Used by :func:`semisup_ae_loss`.
* ``annotation_label_2`` – per-patch integer class from an optional second
                           annotation file (e.g. Position: 0–4), or ``-1``.
                           Used by :func:`semisup_ae_loss_dual`.
* ``path``               – absolute path to the .tif file (str).

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
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
        transform=None,
    ):
        self.root_dir       = root_dir
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.transform      = transform

        # ---- helper: load one annotation file → {filename: int} ----
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        # ---- primary annotation ----
        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])

        # ---- secondary annotation ----
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ---- load images ----
        all_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(("tif", "tiff"))
        ])

        self.data                = []
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for img_path in all_paths:
            try:
                image = tiff.imread(img_path).astype(np.float32)
                if self.transform:
                    image = self.transform(image)
                image = torch.tensor(image, dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Skipping unreadable image {img_path} – {e}")
                continue

            key = _patch_name_to_annotation_key(img_path)
            self.data.append(image)
            self.paths.append(img_path)
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"PatchDataset [{self.condition_name}]: {len(self.data)} patches, "
            f"condition={condition},{ann_info or ' (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
            self.paths[idx],
        )


# ---------------------------------------------------------------------------
# Multi-channel dataset
# ---------------------------------------------------------------------------

class MultiChannelPatchDataset(Dataset):
    """Dataset that stacks single-channel patches from multiple directories.

    Patches are matched by filename across all ``channel_dirs``.  Only filenames
    present in **every** directory are included (intersection).  Matched patches
    are stacked into a ``(C, H, W)`` tensor where ``C = len(channel_dirs)``.

    Returns the same 5-tuple as :class:`PatchDataset`::

        (image, condition, annotation_label, annotation_label_2, path)

    where ``image`` is ``(C, H, W)`` float32 and ``path`` is the first channel's
    file path.  Annotation lookup uses the first channel's filename as the key.

    Parameters
    ----------
    channel_dirs : list[str]
        Ordered list of directories, one per channel (e.g.
        ``[".../tiff_patches32_ch0", ".../tiff_patches32_ch1", ...]``).
        Must contain at least 2 entries.
    condition : int
        Integer condition ID applied uniformly to every patch.
    condition_name : str
        Human-readable condition label (for logging only).
    annotation_file : str or None
        Path to primary annotation CSV/Excel. ``None`` → all labels ``-1``.
    label_col : str
        Column used as primary class label.
    filename_col : str
        Column holding patch filenames in the annotation file.
    label_order : list[str] or None
        Ordered label list for integer mapping. ``None`` → alphabetical sort.
    annotation_file_2 : str or None
        Optional secondary annotation file.
    label_col_2, filename_col_2, label_order_2 :
        Same as above for the secondary annotation.
    transform : callable or None
        Applied to each raw ``(H, W)`` float32 numpy array before stacking.
    """

    def __init__(
        self,
        channel_dirs: list,
        condition: int = 0,
        condition_name: str = "",
        annotation_file: str | None = None,
        label_col: str = "Classification",
        filename_col: str = "crop_img_filename",
        label_order: list | None = None,
        annotation_file_2: str | None = None,
        label_col_2: str = "Position",
        filename_col_2: str = "crop_img_filename",
        label_order_2: list | None = None,
        transform=None,
    ):
        if len(channel_dirs) < 2:
            raise ValueError(
                f"MultiChannelPatchDataset requires at least 2 channel directories, "
                f"got {len(channel_dirs)}."
            )

        self.channel_dirs   = channel_dirs
        self.condition      = condition
        self.condition_name = condition_name or str(condition)
        self.transform      = transform

        # ---- annotation loader (identical logic to PatchDataset) ----
        def _load_annotations(ann_file, col, fname_col, order):
            if not ann_file:
                return {}, order or [], {}, 0
            ann_path = Path(ann_file)
            ann_df   = (pd.read_excel(ann_path)
                        if ann_path.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(ann_path))
            ann_df[fname_col] = ann_df[fname_col].astype(str).apply(lambda p: Path(p).name)
            fname_to_str = dict(zip(ann_df[fname_col], ann_df[col].astype(str)))
            if not order:
                order = sorted({v for v in fname_to_str.values() if v and v != "nan"})
            lbl_to_int   = {lbl: i for i, lbl in enumerate(order)}
            fname_to_int = {f: lbl_to_int.get(s, -1) for f, s in fname_to_str.items()}
            return fname_to_int, order, lbl_to_int, len(order)

        fname_to_ann1, self.label_order, self.label_to_int, self.num_classes = \
            _load_annotations(annotation_file, label_col, filename_col, label_order or [])
        fname_to_ann2, self.label_order_2, self.label_to_int_2, self.num_classes_2 = \
            _load_annotations(annotation_file_2, label_col_2, filename_col_2, label_order_2 or [])

        # ---- find filenames present in every channel directory ----
        def _tif_names(d):
            return {f for f in os.listdir(d) if f.lower().endswith(("tif", "tiff"))}

        common = _tif_names(channel_dirs[0])
        for d in channel_dirs[1:]:
            common &= _tif_names(d)
        common = sorted(common)

        n_missing = sum(
            len(_tif_names(d)) for d in channel_dirs
        ) // len(channel_dirs) - len(common)
        if n_missing > 0:
            print(
                f"MultiChannelPatchDataset: {n_missing} patch(es) dropped "
                f"(not present in all {len(channel_dirs)} channel directories)."
            )

        # ---- load and stack ----
        self.data                = []
        self.paths               = []
        self.annotation_labels   = []
        self.annotation_labels_2 = []

        for fname in common:
            try:
                planes = []
                for d in channel_dirs:
                    ch_arr = tiff.imread(os.path.join(d, fname)).astype(np.float32)
                    if self.transform:
                        ch_arr = self.transform(ch_arr)
                    planes.append(ch_arr)
                # stack → (C, H, W)
                image = torch.tensor(np.stack(planes, axis=0), dtype=torch.float32)
            except Exception as e:
                print(f"Warning: Skipping {fname} – {e}")
                continue

            key = _patch_name_to_annotation_key(fname)
            self.data.append(image)
            self.paths.append(os.path.join(channel_dirs[0], fname))
            self.annotation_labels.append(fname_to_ann1.get(key, -1))
            self.annotation_labels_2.append(fname_to_ann2.get(key, -1))

        n_ann  = sum(1 for l in self.annotation_labels   if l >= 0)
        n_ann2 = sum(1 for l in self.annotation_labels_2 if l >= 0)
        ann_info = ""
        if annotation_file:
            ann_info += f"  label1={label_col}: {n_ann} annotated"
        if annotation_file_2:
            ann_info += f"  label2={label_col_2}: {n_ann2} annotated"
        print(
            f"MultiChannelPatchDataset [{self.condition_name}]: {len(self.data)} patches, "
            f"{len(channel_dirs)} channels, condition={condition}"
            f"{ann_info or ', (unlabelled)'}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return (
            self.data[idx],
            self.condition,
            self.annotation_labels[idx],
            self.annotation_labels_2[idx],
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
        image, condition, _, _, path = self._ds[idx]
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
        image, _, annotation_label, _, path = self._ds[idx]
        return image, annotation_label, path
