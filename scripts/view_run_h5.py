"""
Interactive local viewer for HDF5 run output files produced by pack_run_to_h5.py.

Usage:
    python scripts/view_run_h5.py <run_outputs.h5> [--filter <keyword>]

Navigation:
    - Prints a numbered list of all images in the HDF5
    - Enter a number to display that image
    - Enter a keyword to filter the list
    - Type 'q' to quit

Optional:
    --filter <keyword>  : only show entries whose key contains <keyword>
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 traversal
# ─────────────────────────────────────────────────────────────────────────────

def collect_datasets(hf: h5py.File) -> list[str]:
    """Return sorted list of all dataset keys."""
    keys = []
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append(name)
    hf.visititems(_visit)
    return sorted(keys)


# ─────────────────────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────────────────────

def show_image(key: str, arr: np.ndarray, file_type: str):
    """Display one image with matplotlib. Handles PNG and multi-channel TIFFs."""
    plt.close("all")

    if file_type == "png":
        # arr is H×W×C (RGB/RGBA) or H×W (grey)
        fig, ax = plt.subplots(figsize=(10, 7))
        if arr.ndim == 2 or arr.shape[-1] in (3, 4):
            ax.imshow(arr, cmap="gray" if arr.ndim == 2 else None)
        else:
            ax.imshow(arr[..., 0], cmap="gray")
        ax.set_title(key, fontsize=9, wrap=True)
        ax.axis("off")

    else:
        # TIFF: could be H×W, C×H×W, or H×W×C
        arr_f = arr.astype(np.float32)

        if arr_f.ndim == 2:
            channels = [arr_f]
            ch_names = ["ch0"]
        elif arr_f.ndim == 3:
            # Guess axis: if first dim is small (1-4) treat as C×H×W
            if arr_f.shape[0] <= 4 and arr_f.shape[0] < arr_f.shape[1]:
                channels = [arr_f[i] for i in range(arr_f.shape[0])]
                ch_names = [f"ch{i}" for i in range(arr_f.shape[0])]
            else:
                # H×W×C
                channels = [arr_f[..., i] for i in range(arr_f.shape[2])]
                ch_names = [f"ch{i}" for i in range(arr_f.shape[2])]
        else:
            channels = [arr_f[0, 0]]
            ch_names = ["ch0"]

        n = len(channels)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, ch, name in zip(axes, channels, ch_names):
            vmin, vmax = np.nanpercentile(ch, 1), np.nanpercentile(ch, 99)
            ax.imshow(ch, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        fig.suptitle(key, fontsize=9, wrap=True)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive browser
# ─────────────────────────────────────────────────────────────────────────────

def print_list(keys: list[str], filtered: list[str]):
    print(f"\n{'─'*60}")
    print(f"  {len(filtered)} / {len(keys)} entries")
    print(f"{'─'*60}")
    for i, k in enumerate(filtered):
        print(f"  [{i:3d}]  {k}")
    print(f"{'─'*60}")
    print("  Enter number to view | 'f <word>' to filter | 'r' to reset | 'q' quit")


def run_viewer(h5_path: Path, init_filter: str = ""):
    with h5py.File(h5_path, "r") as hf:
        all_keys = collect_datasets(hf)
        if not all_keys:
            print("No datasets found in HDF5."); return

        current_filter = init_filter.lower()
        filtered = [k for k in all_keys if current_filter in k.lower()] \
                   if current_filter else list(all_keys)

        print(f"\nOpened: {h5_path}  ({len(all_keys)} images total)")
        print_list(all_keys, filtered)

        while True:
            try:
                cmd = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd.lower() == "q":
                break

            elif cmd.lower() == "r":
                current_filter = ""
                filtered = list(all_keys)
                print_list(all_keys, filtered)

            elif cmd.lower().startswith("f "):
                current_filter = cmd[2:].strip().lower()
                filtered = [k for k in all_keys if current_filter in k.lower()]
                print_list(all_keys, filtered)

            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(filtered):
                    key = filtered[idx]
                    ds  = hf[key]
                    arr = ds[()]
                    ft  = ds.attrs.get("file_type", "png")
                    print(f"  Displaying: {key}  shape={arr.shape}  dtype={arr.dtype}")
                    show_image(key, arr, ft)
                else:
                    print(f"  Out of range (0–{len(filtered)-1})")

            else:
                print("  Unknown command. Use a number, 'f <word>', 'r', or 'q'.")

    print("Bye.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browse HDF5 run output files.")
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--filter", default="", help="Initial keyword filter")
    args = parser.parse_args()

    if not args.h5_path.exists():
        print(f"ERROR: {args.h5_path} not found"); sys.exit(1)

    run_viewer(args.h5_path, init_filter=args.filter)


if __name__ == "__main__":
    main()
