"""
Pack all PNG / TIFF outputs from a result directory into a single HDF5 file.

Usage:
    python scripts/pack_run_to_h5.py <result_dir> [--out <output.h5>] [--delete]

Arguments:
    result_dir  : directory to scan recursively (e.g. results/test_run/baseline)
    --out       : output HDF5 path (default: <result_dir>/run_outputs.h5)
    --delete    : delete original files after packing (optional)

Output HDF5 structure mirrors the directory tree:
    recon/visual/train_comparison   <- PNG stored as uint8 H×W×C
    recon/patches/raw_val_patch001  <- TIFF stored as float32 C×H×W or H×W
    ae_recon_ep050                  <- PNG stored as uint8
    ...

Each dataset has attrs:
    filename   : original file name
    file_type  : "png" or "tif"
    rel_path   : relative path from result_dir
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image


EXTS_PNG = {".png"}
EXTS_TIF = {".tif", ".tiff"}
ALL_EXTS  = EXTS_PNG | EXTS_TIF


def read_png(path: Path) -> np.ndarray:
    """Read PNG → uint8 numpy array (H, W, C) or (H, W)."""
    img = Image.open(path)
    return np.array(img)


def read_tif(path: Path) -> np.ndarray:
    """Read TIFF → float32 numpy array."""
    arr = tifffile.imread(str(path))
    return arr.astype(np.float32)


def path_to_key(rel: Path) -> str:
    """Convert relative file path to HDF5 dataset key (no extension)."""
    parts = list(rel.with_suffix("").parts)
    return "/".join(parts)


def pack_directory(result_dir: Path, out_path: Path, delete: bool = False):
    files = sorted(
        p for p in result_dir.rglob("*")
        if p.suffix.lower() in ALL_EXTS and p != out_path
    )

    if not files:
        print(f"No PNG/TIFF files found in {result_dir}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) → packing into {out_path}")

    with h5py.File(out_path, "w") as hf:
        for fpath in files:
            rel = fpath.relative_to(result_dir)
            key = path_to_key(rel)
            ext = fpath.suffix.lower()

            try:
                if ext in EXTS_PNG:
                    arr = read_png(fpath)
                    ds  = hf.create_dataset(key, data=arr,
                                            compression="gzip", compression_opts=4)
                    ds.attrs["file_type"] = "png"
                else:
                    arr = read_tif(fpath)
                    ds  = hf.create_dataset(key, data=arr,
                                            compression="gzip", compression_opts=4)
                    ds.attrs["file_type"] = "tif"

                ds.attrs["filename"] = fpath.name
                ds.attrs["rel_path"] = str(rel)
                print(f"  packed  {rel}  {arr.shape} {arr.dtype}")

            except Exception as e:
                print(f"  SKIP    {rel}  ({e})")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nDone. HDF5 size: {size_mb:.1f} MB → {out_path}")

    if delete:
        for fpath in files:
            fpath.unlink()
        print(f"Deleted {len(files)} original file(s).")


def main():
    parser = argparse.ArgumentParser(description="Pack PNG/TIFF run outputs into HDF5.")
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output HDF5 path (default: result_dir/run_outputs.h5)")
    parser.add_argument("--delete", action="store_true",
                        help="Delete original files after packing")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        print(f"ERROR: {result_dir} is not a directory"); sys.exit(1)

    out_path = args.out.resolve() if args.out else result_dir / "run_outputs.h5"
    pack_directory(result_dir, out_path, delete=args.delete)


if __name__ == "__main__":
    main()
