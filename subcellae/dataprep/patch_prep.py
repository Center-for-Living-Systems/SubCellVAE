"""
patch_prep.py
=============
Primitives for patch extraction from microscopy images.

Supported file types
--------------------
``"czi"``  – Zeiss .czi files (loaded via *czifile*).  Raw pixel values are
             divided by ``255 × 255`` to bring them into a nominal ``[0, 1]``
             floating-point range.
``"npy"``  – NumPy .npy files.  Expected array shape after ``np.load`` +
             ``squeeze``: either ``(C, H, W)`` (multi-channel) or ``(H, W)``
             (single-channel / already-selected channel).  Values are used
             as-is (no ``/ 255²`` rescaling); apply normalization if needed.

The file type is controlled by the ``file_type`` parameter of
:func:`list_image_files`, :func:`compute_dataset_norm_stats`, and
:func:`load_and_pad`.

Normalization options
---------------------
Two optional normalization modes are available, applied **after** raw loading
and **before** padding:

1. ``"dataset"``  – Whole-dataset, per-channel 1 %–99 % percentile stretch.
   Call :func:`compute_dataset_norm_stats` once across all files in a run to
   collect ``(p1, p99)`` per channel, then pass the resulting dict as
   ``norm_stats`` to :func:`load_and_pad`.

2. ``"image"``    – Per-image, per-channel 1 %–99 % percentile stretch.
   Stats are computed on the fly inside :func:`load_and_pad` from the single
   image being loaded; no pre-computation required.

Pass ``norm_mode=None`` (default) to skip normalization entirely.  For .czi
files without normalization the original ``/ (255 × 255)`` scaling is still
applied.  For .npy files without normalization values are used unchanged.

Segmentation options
--------------------
Segmentation in ``load_and_pad`` is automatic:

* If *cell_mask_folder* is provided **and** the directory is non-empty, a
  pre-computed mask .tif is read from that folder
  (naming: ``cell_mask_<filename>.tif``).
* Otherwise (folder is ``None`` or empty) :func:`segment_cell_mask` is called
  on a chosen channel of the loaded image at run-time.
  Controlled by ``seg_ch``, ``seg_threshold``, and ``seg_close_size``.
"""

import math
import os
import random
from typing import Dict, Literal, Optional, Tuple

import czifile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import tifffile
from skimage import transform
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing, binary_opening, disk, remove_small_objects,
)
from skimage.restoration import rolling_ball


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# {channel_index: (p1_value, p99_value)}
NormStats = Dict[int, Tuple[float, float]]

NormMode = Optional[Literal["dataset", "image", "cell_insideoutside", "cell_minmax"]]

FileType = Literal["czi", "npy"]

SegMode = Literal["file", "on_the_fly"]

_EXT_MAP: Dict[str, str] = {
    "czi": ".czi",
    "npy": ".npy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_padding(input_img: np.ndarray, pad_size: int, value: float) -> np.ndarray:
    """Zero-pad (or constant-pad) a 2-D array on all four sides."""
    output_img = (
        np.zeros([input_img.shape[0] + pad_size * 2,
                  input_img.shape[1] + pad_size * 2]) + value
    )
    output_img[
        pad_size: input_img.shape[0] + pad_size,
        pad_size: input_img.shape[1] + pad_size,
    ] = input_img
    return output_img


def rotate_coor(
    x_i: np.ndarray, y_i: np.ndarray,
    x_c: float, y_c: float,
    rotate_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates (x_i, y_i) around centre (x_c, y_c)."""
    rotate_angle = rotate_angle * np.pi / 180
    x_o = (
        (x_i - x_c) * math.cos(rotate_angle)
        - (2 * y_c - y_i - y_c) * math.sin(rotate_angle)
        + x_c
    )
    y_o = (
        -(x_i - x_c) * math.sin(rotate_angle)
        - (2 * y_c - y_i - y_c) * math.cos(rotate_angle)
        + (2 * y_c - y_c)
    )
    return x_o, y_o


def list_image_files(image_folder: str, file_type: FileType = "czi") -> list:
    """Return a sorted list of image filenames in *image_folder*.

    Parameters
    ----------
    image_folder:
        Directory to scan.
    file_type:
        ``"czi"`` – return only ``.czi`` files (default, original behaviour).
        ``"npy"`` – return only ``.npy`` files.

    Returns
    -------
    list[str]
        Sorted basenames (no directory path).
    """
    if file_type not in _EXT_MAP:
        raise ValueError(
            f"file_type must be one of {list(_EXT_MAP)}, got {file_type!r}"
        )
    ext = _EXT_MAP[file_type]
    return sorted([
        x for x in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, x)) and x.endswith(ext)
    ])


# backward-compatible alias
def list_czi_files(image_folder: str) -> list:
    """Deprecated alias for ``list_image_files(folder, file_type='czi')``."""
    return list_image_files(image_folder, file_type="czi")


# ---------------------------------------------------------------------------
# Private loading helpers (file-type aware)
# ---------------------------------------------------------------------------

def _load_raw_squeezed(
    image_folder: str,
    filename: str,
    file_type: FileType,
) -> np.ndarray:
    """Load and squeeze a single image file; return a float array.

    For ``"czi"`` files values are divided by ``255 × 255``.
    For ``"npy"`` files values are used as-is (already float or will be cast).

    The returned array is either ``(C, H, W)`` (multi-channel) or ``(H, W)``
    (single-channel / pre-selected).
    """
    path = os.path.join(image_folder, filename)
    if file_type == "czi":
        raw = czifile.imread(path).squeeze().astype(float) / (255 * 255)
    elif file_type == "npy":
        raw = np.load(path).squeeze().astype(float)
    else:
        raise ValueError(f"Unsupported file_type {file_type!r}. Choose 'czi' or 'npy'.")
    return raw


def _extract_channel(
    raw: np.ndarray,
    channel: int,
    filename: str,
    file_type: FileType,
) -> np.ndarray:
    """Extract a single 2-D channel plane from a raw squeezed array.

    Parameters
    ----------
    raw:
        Squeezed float array; shape ``(C, H, W)`` or ``(H, W)``.
    channel:
        Channel index to extract (0-based).
    filename:
        Used only for error messages.
    file_type:
        Used only for error messages.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(H, W)``.
    """
    if raw.ndim == 3:
        return raw[channel, :, :]
    elif raw.ndim == 2:
        # single-channel – channel index must be 0
        if channel != 0:
            raise IndexError(
                f"{file_type} file {filename!r} is single-channel (shape "
                f"{raw.shape}) but channel={channel} was requested."
            )
        return raw
    else:
        raise ValueError(
            f"Unexpected squeezed array shape {raw.shape} from {filename!r}."
        )


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

def _percentile_stretch(
    img: np.ndarray,
    p1: float,
    p99: float,
) -> np.ndarray:
    """Clip and linearly rescale *img* to [0, 1] using provided percentile bounds.

    Parameters
    ----------
    img:
        Float array (any range).
    p1, p99:
        Lower and upper clipping bounds (e.g. 1st and 99th percentiles).

    Returns
    -------
    np.ndarray
        Array clipped to ``[p1, p99]`` and rescaled to ``[0, 1]``.
        If ``p99 == p1`` the array is returned as all zeros to avoid
        divide-by-zero.
    """
    if p99 == p1:
        return np.zeros_like(img)
    out = np.clip(img, p1, p99)
    out = (out - p1) / (p99 - p1)
    return out


def compute_dataset_norm_stats(
    image_folder: str,
    filenames: list,
    channels: list,
    lo: float = 1.0,
    hi: float = 99.0,
    file_type: FileType = "czi",
    rolling_ball_radius: Optional[float] = None,
) -> NormStats:
    """Compute per-channel percentile bounds over the **whole dataset**.

    Loads every file listed in *filenames*, reads the requested *channels*,
    pools all pixel values per channel, then returns the ``lo``/``hi``
    percentile pair for each channel.

    If *rolling_ball_radius* is set, rolling-ball background subtraction is
    applied to each channel before pooling pixels, so the percentiles are
    computed on the same signal that will be normalised during patch extraction.

    Parameters
    ----------
    image_folder:
        Directory containing the image files.
    filenames:
        Ordered list of image basenames (as returned by
        :func:`list_image_files`).
    channels:
        List of integer channel indices to compute stats for
        (e.g. ``[0, 1, 2, 3]``).
    lo, hi:
        Percentile values (0–100). Defaults: 1 and 99.
    file_type:
        ``"czi"`` – load via *czifile*; values divided by ``255 × 255``.
        ``"npy"`` – load via ``np.load``; values used as-is.
    rolling_ball_radius:
        If provided, apply rolling-ball subtraction to each channel before
        computing percentiles. Must match the value used during patch extraction.

    Returns
    -------
    NormStats
        ``{channel_index: (p_lo, p_hi), ...}``
    """
    pixel_pools: Dict[int, list] = {ch: [] for ch in channels}

    for filename in filenames:
        raw = _load_raw_squeezed(image_folder, filename, file_type)
        for ch in channels:
            ch_data = _extract_channel(raw, ch, filename, file_type)
            if rolling_ball_radius is not None:
                # CZI images are divided by 255² at load time (values ~0–1).
                # Rolling ball radius is in intensity units, so we must work at
                # uint16 scale (radius=20 is meaningful there, not on 0–1 values).
                if file_type == "czi":
                    _scale = 255.0 * 255.0
                    ch_data = apply_rolling_ball(ch_data * _scale, radius=rolling_ball_radius) / _scale
                else:
                    ch_data = apply_rolling_ball(ch_data, radius=rolling_ball_radius)
            pixel_pools[ch].append(ch_data.ravel())

    norm_stats: NormStats = {}
    for ch in channels:
        all_pixels = np.concatenate(pixel_pools[ch])
        norm_stats[ch] = (
            float(np.percentile(all_pixels, lo)),
            float(np.percentile(all_pixels, hi)),
        )

    return norm_stats


def normalize_image(
    img: np.ndarray,
    channel: int,
    norm_mode: NormMode,
    norm_stats: Optional[NormStats] = None,
    lo: float = 1.0,
    hi: float = 99.0,
) -> np.ndarray:
    """Apply the chosen normalization to a single 2-D channel image.

    Parameters
    ----------
    img:
        2-D float array already divided by ``255 * 255``.
    channel:
        Channel index – used to look up pre-computed stats in *norm_stats*.
    norm_mode:
        ``"dataset"``  → use *norm_stats* (must be provided).
        ``"image"``    → compute percentiles on-the-fly from *img*.
        ``None``       → return *img* unchanged.
    norm_stats:
        Required when ``norm_mode == "dataset"``.
    lo, hi:
        Percentile bounds used for ``"image"`` mode. Ignored for ``"dataset"``
        mode (bounds were fixed at :func:`compute_dataset_norm_stats` call
        time).

    Returns
    -------
    np.ndarray
        Normalised (or unchanged) image.
    """
    if norm_mode is None:
        return img

    if norm_mode == "dataset":
        if norm_stats is None:
            raise ValueError(
                "norm_stats must be provided when norm_mode='dataset'."
            )
        if channel not in norm_stats:
            raise KeyError(
                f"Channel {channel} not found in norm_stats "
                f"(available: {list(norm_stats.keys())})."
            )
        p1, p99 = norm_stats[channel]
        return _percentile_stretch(img, p1, p99)

    if norm_mode == "image":
        p1  = float(np.percentile(img, lo))
        p99 = float(np.percentile(img, hi))
        return _percentile_stretch(img, p1, p99)

    raise ValueError(
        f"Unknown norm_mode {norm_mode!r}. "
        f"Choose 'dataset', 'image', 'cell_insideoutside', 'cell_minmax', or None."
    )


# ---------------------------------------------------------------------------
# On-the-fly segmentation
# ---------------------------------------------------------------------------

def _correct_seg_illumination(cellmask_img: np.ndarray) -> np.ndarray:
    """Gaussian flat-field correction for the segmentation channel.

    The original pre-computed masks were generated with this correction applied
    first, so on-the-fly segmentation must replicate it to produce comparable
    cell outlines.

    Models illumination as a Gaussian hill centred in the frame (slightly
    brighter at centre — typical epi-fluorescence vignetting), then divides it
    out so that the subsequent threshold is uniform across the field of view.

    Background model: ``gauss(sigma=1 on [-1,1]) * 8 + 100``
      - centre value: 108  (1.0 * 8 + 100)
      - edge value  : ~100 (0.0 * 8 + 100)
      → ~8 % centre-to-edge correction
    """
    H, W = cellmask_img.shape
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing='ij')
    gauss = np.exp(-((x ** 2 + y ** 2) / 2.0))   # sigma=1 on normalised [-1,1] grid
    background = gauss * 8 + 100

    smoothed = ndi.gaussian_filter(cellmask_img.astype(float), sigma=2,
                                   mode='nearest', truncate=3)
    return smoothed * 100.0 / background


def segment_cell_mask(
    cellmask_img: np.ndarray,
    threshold: float = 0.2,
    close_size: int = 5,
    min_size_initial: int = 3,
    min_size_post_close: int = 10,
    min_size_final: int = 30000,
) -> np.ndarray:
    """Segment a cell-body mask from a single 2-D intensity image.

    The pipeline mirrors the original ``target_cell_mask_seg`` class method,
    extracted as a pure function with all previously hard-coded / instance
    parameters exposed as arguments.

    Steps
    -----
    0. Gaussian flat-field correction (:func:`_correct_seg_illumination`):
       models centre-bright vignetting and divides it out so the threshold
       is spatially uniform.  Matches the correction applied when the
       pre-computed masks were generated.
    1. If the corrected image is not already in [0, 1], apply a per-image
       1 %–99 % percentile stretch; otherwise use the values as-is.
    2. Threshold at *threshold*.
    3. Remove small objects (< *min_size_initial* px).
    4. Binary closing with a disk of radius *close_size*.
    5. Remove small objects (< *min_size_post_close* px).
    6. Binary opening with a disk of radius 3.
    7. Fill holes.
    8. Remove objects smaller than *min_size_final* px (keeps whole-cell blobs).
    9. If more than one connected component remains, keep the one closest to
       the image centre; also keep the second-closest if it does not touch any
       image border.

    Parameters
    ----------
    cellmask_img:
        2-D float array. If values are outside [0, 1], a per-image 1 %–99 %
        percentile stretch is applied; already-normalized data is used as-is.
    threshold:
        Binarization threshold applied after 1 %–99 % normalization.
        Default ``0.2``.
    close_size:
        Radius (px) of the disk structuring element used in binary closing.
        Default ``5``.
    min_size_initial:
        Minimum object size (px²) after initial thresholding. Default ``3``.
    min_size_post_close:
        Minimum object size (px²) after closing. Default ``10``.
    min_size_final:
        Minimum object size (px²) after hole-filling; removes small debris and
        keeps only whole-cell bodies. Default ``30000``.

    Returns
    -------
    np.ndarray
        Integer label image (same shape as *cellmask_img*). Background = 0;
        retained cell regions carry their original label values (≥ 1).
    """
    # --- step 0: flat-field / illumination correction ---
    img_float = _correct_seg_illumination(cellmask_img)

    # --- step 1: always percentile-stretch to [0, 1] ---
    normalized = _percentile_stretch(
        img_float,
        p1=float(np.percentile(img_float, 1)),
        p99=float(np.percentile(img_float, 99)),
    )


    # --- step 2: threshold ---
    mask = normalized > threshold

    # --- step 3: remove tiny specks ---
    mask = remove_small_objects(mask, min_size=min_size_initial, connectivity=1)

    # --- step 4: close gaps ---
    mask = binary_closing(mask, disk(close_size))

    # --- step 5: remove small post-close fragments ---
    mask = remove_small_objects(mask, min_size=min_size_post_close, connectivity=1)

    # --- step 6: open (remove thin protrusions) ---
    mask = binary_opening(mask, disk(3))

    # --- step 7: fill holes ---
    mask = ndi.binary_fill_holes(mask)

    # --- step 8: remove sub-cellular debris, keep whole-cell bodies ---
    mask = remove_small_objects(mask, min_size=min_size_final, connectivity=1)

    # --- step 9: keep centre cell (+ optional second cell if off-border) ---
    label_img = label(mask)
    regions = regionprops(label_img)

    if len(regions) <= 1:
        # zero or one region – nothing to filter
        return label_img

    H, W = mask.shape
    cx, cy = H / 2.0, W / 2.0

    # distance of each region centroid from image centre
    distances = np.array([
        abs(r.centroid[0] - cx) + abs(r.centroid[1] - cy)
        for r in regions
    ])

    # flag regions that touch any image border
    on_border = np.array([
        (r.bbox[0] == 0 or r.bbox[1] == 0
         or r.bbox[2] == H or r.bbox[3] == W)
        for r in regions
    ], dtype=bool)

    sort_idx = np.argsort(distances)
    first_region  = regions[sort_idx[0]]
    second_region = regions[sort_idx[1]]

    out = np.zeros_like(label_img)
    out[label_img == first_region.label] = first_region.label

    # include second-closest only if it does not touch the border
    if not on_border[sort_idx[1]]:
        out[label_img == second_region.label] = second_region.label

    return out


# ---------------------------------------------------------------------------
# Cell inside/outside normalization
# ---------------------------------------------------------------------------

def normalize_cell_insideoutside(
    img: np.ndarray,
    seg: np.ndarray,
    scale: float = 5.0,
) -> np.ndarray:
    """Normalize using the cell mask to separate background from signal.

    Steps
    -----
    1. ``mean_outside = mean(pixels where seg == 0)``
    2. ``int1 = img - mean_outside``
    3. ``denom = mean(img where seg > 0) - mean_outside``
    4. Return ``int1 / (denom * scale)``  (no clipping — matches coworker formula)

    Parameters
    ----------
    img:
        2-D float image (unpadded, same shape as *seg*).
    seg:
        2-D segmentation mask (unpadded). Zero = outside cell, nonzero = inside.
    scale:
        Constant divisor applied after step 3. Default ``5.0``.

    Returns
    -------
    np.ndarray
        Normalised image (unclipped; values may be slightly negative or >1).
    """
    outside = seg == 0
    inside  = seg > 0

    mean_outside = float(np.mean(img[outside])) if outside.any() else 0.0
    mean_inside  = float(np.mean(img[inside]))  if inside.any()  else mean_outside + 1.0

    int1  = img - mean_outside
    denom = mean_inside - mean_outside
    if denom == 0.0:
        denom = 1.0

    return int1 / (denom * scale)


def normalize_cell_minmax(
    img: np.ndarray,
    seg: np.ndarray,
) -> np.ndarray:
    """Normalize using mean background subtraction, cell mean division, then max stretch.

    Steps
    -----
    1. ``int1 = img - mean(pixels where seg == 0)``   — subtract background mean
    2. ``int2 = int1 / mean(int1 where seg > 0)``     — scale by mean in-cell signal
    3. Return ``int2 / int2.max()``                   — stretch to [0, 1]

    Matches the preprocessing notebook (Alana): ``img_pp = (img_RB - avg_BG) / avg_cell``
    followed by ``img_pp_norm = img_pp / img_pp.max()``.

    Parameters
    ----------
    img:
        2-D float image (unpadded, same shape as *seg*).
    seg:
        2-D segmentation mask (unpadded). Zero = outside cell, nonzero = inside.

    Returns
    -------
    np.ndarray
        Normalised image clipped to [0, 1] with max value = 1.
    """
    outside = seg == 0
    inside  = seg > 0

    bg = float(np.mean(img[outside])) if outside.any() else 0.0
    int1 = img - bg

    # Normalise by the 99th percentile of in-cell signal so that the typical
    # bright FA maps to ~1 rather than the single hottest pixel driving
    # everything else into the dim end of [0, 1].
    if inside.any():
        scale = float(np.percentile(int1[inside], 99))
    else:
        scale = float(int1.max())
    if scale <= 0.0:
        scale = 1.0

    return np.clip(int1 / scale, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Rolling-ball background subtraction
# ---------------------------------------------------------------------------

def apply_rolling_ball(img: np.ndarray, radius: float) -> np.ndarray:
    """Subtract a rolling-ball background estimate from *img*.

    Uses :func:`skimage.restoration.rolling_ball` to estimate the slowly-
    varying background, then subtracts it and clips the result to ``[0, 1]``.
    Applied per 2-D channel image **before** normalization.

    Parameters
    ----------
    img:
        2-D float array (values in nominal ``[0, 1]`` range after raw load).
    radius:
        Radius of the rolling ball in pixels.  Larger values remove broader
        background variations.  Typical range: 20–200 px.

    Returns
    -------
    np.ndarray
        Background-subtracted image. Negative values (over-subtracted background)
        are left as-is so that downstream normalization (e.g. percentile stretch)
        can map the full range naturally without introducing exact zeros.
    """
    background = rolling_ball(img, radius=radius)
    return img - background


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_and_pad(
    image_folder: str,
    cell_mask_folder: Optional[str],
    filename: str,
    major_ch: int,
    pad_size: int = 64,
    img_pad_val: Optional[float] = None,
    norm_mode: NormMode = None,
    norm_stats: Optional[NormStats] = None,
    norm_lo: float = 1.0,
    norm_hi: float = 99.0,
    file_type: FileType = "czi",
    seg_ch: Optional[int] = None,
    seg_threshold: float = 0.2,
    seg_close_size: int = 5,
    seg_min_size_initial: int = 3,
    seg_min_size_post_close: int = 10,
    seg_min_size_final: int = 30000,
    rolling_ball_radius: Optional[float] = None,
    norm_cell_scale: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image + cell mask, optionally normalize, then pad both.

    Parameters
    ----------
    image_folder:
        Directory containing the raw image files.
    cell_mask_folder:
        Directory containing pre-computed cell-mask .tif files
        (naming: ``cell_mask_<filename>.tif``).
        If ``None`` or the directory is empty, on-the-fly segmentation is used.
    filename:
        Basename of the image file (e.g. ``"myfile.czi"`` or ``"myfile.npy"``).
    major_ch:
        Channel index to extract for the image patch (0-based).
    pad_size:
        Pixels of constant padding added to each edge. Default ``64``.
    img_pad_val:
        Constant used to pad the image. Defaults to the image mean.
    norm_mode:
        Normalization strategy:
        ``None``        – no normalization.
        ``"dataset"``   – whole-dataset 1 %–99 % stretch (requires *norm_stats*).
        ``"image"``     – per-image 1 %–99 % stretch computed on the fly.
    norm_stats:
        Pre-computed ``{channel: (p1, p99)}`` dict. Required when
        ``norm_mode='dataset'``, ignored otherwise.
    norm_lo, norm_hi:
        Percentile bounds for ``"image"`` mode. Defaults: 1 and 99.
    file_type:
        ``"czi"`` – load via *czifile* (default).
        ``"npy"`` – load via ``np.load``.
    seg_ch:
        Channel index used as input to :func:`segment_cell_mask` for on-the-fly
        segmentation. Defaults to *major_ch* when ``None``.
    seg_threshold:
        Binarization threshold for on-the-fly segmentation. Default ``0.2``.
    seg_close_size:
        Closing disk radius (px) for on-the-fly segmentation. Default ``5``.
    seg_min_size_initial:
        Minimum object size (px²) after initial thresholding. Default ``3``.
    seg_min_size_post_close:
        Minimum object size (px²) after closing. Default ``10``.
    seg_min_size_final:
        Minimum object size (px²) to keep whole-cell bodies. Default ``30000``.

    Returns
    -------
    img : np.ndarray
        Padded (and optionally normalized) float image.
    seg : np.ndarray
        Padded segmentation mask (float).
    """
    # --- raw load (file-type aware) ---
    raw = _load_raw_squeezed(image_folder, filename, file_type)
    img = _extract_channel(raw, major_ch, filename, file_type)

    # --- rolling-ball background subtraction (before normalization) ---
    if rolling_ball_radius is not None:
        # CZI images are divided by 255² at load time (values ~0–1).
        # Rolling ball radius is in intensity units, so work at uint16 scale.
        if file_type == "czi":
            _scale = 255.0 * 255.0
            img = apply_rolling_ball(img * _scale, radius=rolling_ball_radius) / _scale
        else:
            img = apply_rolling_ball(img, radius=rolling_ball_radius)

    # --- segmentation mask ---
    # Use pre-computed mask if folder is given and non-empty; otherwise segment on the fly.
    _use_file_mask = (
        cell_mask_folder is not None
        and os.path.isdir(cell_mask_folder)
        and any(os.scandir(cell_mask_folder))
    )
    if _use_file_mask:
        seg = tifffile.imread(
            os.path.join(cell_mask_folder, "cell_mask_" + filename + ".tif")
        ).squeeze().astype(float)
    else:
        _seg_ch = seg_ch if seg_ch is not None else major_ch
        seg_input = _extract_channel(raw, _seg_ch, filename, file_type)
        seg = segment_cell_mask(
            seg_input,
            threshold=seg_threshold,
            close_size=seg_close_size,
            min_size_initial=seg_min_size_initial,
            min_size_post_close=seg_min_size_post_close,
            min_size_final=seg_min_size_final,
        ).astype(float)

    # --- normalization (applied to image channel only) ---
    if norm_mode == "cell_insideoutside":
        img = normalize_cell_insideoutside(img, seg, scale=norm_cell_scale)
    elif norm_mode == "cell_minmax":
        img = normalize_cell_minmax(img, seg)
    else:
        img = normalize_image(
            img,
            channel=major_ch,
            norm_mode=norm_mode,
            norm_stats=norm_stats,
            lo=norm_lo,
            hi=norm_hi,
        )

    # --- padding ---
    if img_pad_val is None:
        img_pad_val = float(np.mean(img))

    img = image_padding(img, pad_size, img_pad_val)
    seg = image_padding(seg, pad_size, 0)

    return img, seg


# ---------------------------------------------------------------------------
# Debug figure
# ---------------------------------------------------------------------------

def init_debug_fig(
    train_img: np.ndarray,
    train_seg: np.ndarray,
    dpi: int = 256,
):
    fig, ax = plt.subplots(
        1, 2, figsize=(15.6, 7.8), dpi=dpi, facecolor="w", edgecolor="k"
    )
    ax[0].imshow(train_img, cmap=plt.cm.gray, vmax=1, vmin=0)
    ax[1].imshow(train_seg, cmap=plt.cm.gray, vmax=1, vmin=0)
    return fig, ax


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def compute_grid(
    train_img_shape: Tuple[int, int],
    patch_size: int,
    offset_frac_x: float = 0.0,
    offset_frac_y: float = 0.0,
) -> Tuple[int, int, int, int]:
    """Return ``(x_num, y_num, x_0, y_0)`` for a centred regular grid."""
    H, W = train_img_shape
    x_num = int(np.floor(W / patch_size))
    y_num = int(np.floor(H / patch_size))
    x_0 = int((W - x_num * patch_size) / 2 + offset_frac_x * patch_size)
    y_0 = int((H - y_num * patch_size) / 2 + offset_frac_y * patch_size)
    return x_num, y_num, x_0, y_0


def iter_grid_centers(
    x_num: int, y_num: int,
    x_0: int, y_0: int,
    patch_size: int,
):
    """Yield ``(x_i, y_i, x_c, y_c)`` for every grid position."""
    for x_i in range(x_num):
        for y_i in range(y_num):
            y_c = int(y_0 + (y_i - 0.5) * patch_size)
            x_c = int(x_0 + (x_i - 0.5) * patch_size)
            yield x_i, y_i, x_c, y_c


# ---------------------------------------------------------------------------
# Patch extraction helpers
# ---------------------------------------------------------------------------

def extract_big_patch(
    train_img: np.ndarray,
    train_seg: np.ndarray,
    x_c: int,
    y_c: int,
    double_ps: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
    """Extract a ``(2*double_ps) × (2*double_ps)`` patch centred at (x_c, y_c).

    Returns ``None`` if the patch would exceed image bounds.
    """
    y_left  = y_c - double_ps
    x_left  = x_c - double_ps
    y_right = y_c + double_ps
    x_right = x_c + double_ps

    if (
        y_left < 0 or x_left < 0
        or y_right >= train_img.shape[0]
        or x_right >= train_img.shape[1]
    ):
        return None

    patch_img = train_img[y_left:y_right, x_left:x_right]
    patch_seg = train_seg[y_left:y_right, x_left:x_right]
    return patch_img, patch_seg, x_left, y_left


def apply_optional_translation(
    rand_trans_flag: bool,
    max_shift_px: int = 0,
) -> Tuple[int, int]:
    """Return ``(rand_tx, rand_ty)``. Both are 0 when *rand_trans_flag* is False."""
    if rand_trans_flag:
        rand_tx = random.randint(-max_shift_px, max_shift_px)
        rand_ty = random.randint(-max_shift_px, max_shift_px)
    else:
        rand_tx, rand_ty = 0, 0
    return rand_tx, rand_ty


def first_crop_from_big(
    patch_img: np.ndarray,
    patch_seg: np.ndarray,
    patch_size: int,
    double_ps: int,
    rand_tx: int,
    rand_ty: int,
):
    """Crop a region (still larger than the final patch) with optional translation."""
    cx_left_1  = patch_size - rand_tx
    cx_right_1 = double_ps + patch_size - rand_tx
    cy_up_1    = patch_size - rand_ty
    cy_down_1  = double_ps + patch_size - rand_ty

    big_crop_img = patch_img[cy_up_1:cy_down_1, cx_left_1:cx_right_1]
    big_crop_seg = patch_seg[cy_up_1:cy_down_1, cx_left_1:cx_right_1]

    first_crop_x = np.array([cx_left_1, cx_left_1, cx_right_1, cx_right_1, cx_left_1])
    first_crop_y = np.array([cy_up_1,   cy_down_1, cy_down_1,  cy_up_1,    cy_up_1])

    return big_crop_img, big_crop_seg, (cx_left_1, cy_up_1), first_crop_x, first_crop_y


def apply_optional_rotation(
    big_crop_img: np.ndarray,
    big_crop_seg: np.ndarray,
    rand_rota_flag: bool,
    max_angle_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Rotate image and seg by the same angle. No-op when *rand_rota_flag* is False."""
    if rand_rota_flag:
        rand_angle = (random.random() * 2 - 1) * max_angle_deg
    else:
        rand_angle = 0.0

    if rand_angle == 0:
        return big_crop_img, big_crop_seg, rand_angle

    rot_img = transform.rotate(
        big_crop_img, rand_angle, resize=False, mode="constant", cval=0, clip=True
    )
    rot_seg = transform.rotate(
        big_crop_seg, rand_angle, resize=False, mode="constant", cval=0, clip=True
    )
    return rot_img, rot_seg, rand_angle


def center_crop(
    rot_img: np.ndarray,
    rot_seg: np.ndarray,
    patch_size: int,
    half_ps: int,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Take the final *patch_size* crop from the centre of the rotated big crop."""
    cx_left_2  = half_ps
    cx_right_2 = patch_size + half_ps
    cy_up_2    = half_ps
    cy_down_2  = patch_size + half_ps

    crop_img = rot_img[cy_up_2:cy_down_2, cx_left_2:cx_right_2]
    crop_seg = rot_seg[cy_up_2:cy_down_2, cx_left_2:cx_right_2]
    return crop_img, crop_seg, (cx_left_2, cy_up_2)


def compute_final_polygon_in_full_image(
    patch_size: int,
    rand_angle: float,
    cx_left_2: int,
    cy_up_2: int,
    x_left: int,
    y_left: int,
    cx_left_1: int,
    cy_up_1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return polygon corners (in full-image coords) of the final crop after inverse-rotation."""
    X = np.array([cx_left_2, cx_left_2, cx_left_2 + patch_size, cx_left_2 + patch_size, cx_left_2])
    Y = np.array([cy_up_2,   cy_up_2 + patch_size, cy_up_2 + patch_size, cy_up_2,         cy_up_2])

    X_inv, Y_inv = rotate_coor(X, Y, patch_size, patch_size, -rand_angle)

    X_full = X_inv + x_left + cx_left_1
    Y_full = Y_inv + y_left + cy_up_1
    return X_full, Y_full


# ---------------------------------------------------------------------------
# Distance-to-boundary features
# ---------------------------------------------------------------------------

def distance_to_boundary_features(
    cell_mask: np.ndarray,
    row: int,
    col: int,
    n_orientations: int = 8,
) -> np.ndarray:
    """Compute rotation-invariant distance-to-boundary features.

    Casts rays in *n_orientations* evenly-spaced directions from ``(row, col)``
    within *cell_mask* and records the distance (in pixels) to the first zero
    pixel or image edge in each direction.

    The distances are then **cyclically shifted** so that the direction with the
    smallest distance is placed first.  This makes the feature vector invariant
    to the absolute orientation of the patch within the cell.

    Parameters
    ----------
    cell_mask : np.ndarray
        2-D mask; non-zero values are treated as *inside* the cell.
    row, col : int
        Pixel coordinate of the query point (row-major, i.e. y then x).
    n_orientations : int
        Number of evenly-spaced ray directions to cast. Default ``8``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_orientations,)``; the first element
        corresponds to the direction of minimum distance (rotation-invariant).
    """
    h, w = cell_mask.shape

    # Snap outside-mask point to nearest foreground pixel
    if cell_mask[row, col] == 0:
        foreground = np.argwhere(cell_mask > 0)
        if len(foreground) == 0:
            return np.zeros(n_orientations)
        nearest = foreground[
            np.argmin(np.linalg.norm(foreground - np.array([row, col]), axis=1))
        ]
        row, col = int(nearest[0]), int(nearest[1])

    angles_deg = np.linspace(0, 360, n_orientations, endpoint=False)
    distances = np.zeros(n_orientations)

    for i, angle_deg in enumerate(angles_deg):
        angle_rad = np.deg2rad(angle_deg)
        dr = np.sin(angle_rad)
        dc = np.cos(angle_rad)

        step = 0.0
        while True:
            step += 0.5
            ri = int(round(row + dr * step))
            ci = int(round(col + dc * step))

            if ri < 0 or ri >= h or ci < 0 or ci >= w:
                distances[i] = step
                break
            if cell_mask[ri, ci] == 0:
                distances[i] = step
                break

    # Cyclic shift so minimum-distance direction is first → rotation invariant
    min_idx = int(np.argmin(distances))
    return np.roll(distances, -min_idx)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_patch(
    movie_partitioned_data_dir: str,
    crop_img_filename: str,
    crop_patch_img: np.ndarray,
) -> None:
    tifffile.imwrite(
        os.path.join(movie_partitioned_data_dir, crop_img_filename),
        crop_patch_img.astype(np.float32),
        imagej=True,
        metadata={"axes": "YX"},
    )


def make_record_row(
    image_folder: str,
    filename: str,
    filenameID: int,
    x_c: int,
    y_c: int,
    rand_angle: float,
    rand_tx: int,
    rand_ty: int,
    X_full: np.ndarray,
    Y_full: np.ndarray,
    movie_partitioned_data_dir: str,
    crop_img_filename: str,
    movie_plot_dir: str,
    plot_filename: str,
) -> pd.Series:
    return pd.Series(
        [
            image_folder, filename, filenameID, x_c, y_c,
            rand_angle, rand_tx, rand_ty,
            X_full[0], X_full[1], X_full[2], X_full[3],
            Y_full[0], Y_full[1], Y_full[2], Y_full[3],
            movie_partitioned_data_dir, crop_img_filename,
            movie_plot_dir, plot_filename,
        ],
        index=[
            "image_folder", "filename", "filenameID", "x_c", "y_c",
            "rand_angle", "rand_tx", "rand_ty",
            "x_corner1", "x_corner2", "x_corner3", "x_corner4",
            "y_corner1", "y_corner2", "y_corner3", "y_corner4",
            "movie_partitioned_data_dir", "crop_img_filename",
            "movie_plot_dir", "plot_filename",
        ],
    )
