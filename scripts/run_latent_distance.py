#!/usr/bin/env python3
"""
Compute latent-space clustering quality metrics for a run directory.

For each variant subdir that contains a latents.csv, produces:
  latent_dist_condition.png          — control vs ycomp
  latent_dist_fa_type.png            — FA type labels  (from fa_cls_*/predictions_all.csv)
  latent_dist_position.png           — position labels (from pos_cls_*/predictions_all.csv)
  latent_dist_fa_pos_combined.png    — FA type × position combined
  latent_dist_annotation.png         — annotation_label_name col (if present in latents.csv)
  recon_scatter.png                  — pixel-level input vs reconstruction hexbin
  {run_dir}/latent_distance_summary.csv

Metrics computed per label type:
  • intra/inter distance ratio  (higher = better separation)
  • silhouette score            (range -1..1, higher = better)
  • Calinski-Harabasz index     (higher = better)

Reconstruction quality:
  • Pearson r between input and reconstructed pixel intensities (per condition + overall)

Usage:
  python scripts/run_latent_distance.py <run_dir>
  python scripts/run_latent_distance.py <run_dir> --metric cosine
  python scripts/run_latent_distance.py <run_dir> --sample-pairs 30000
  python scripts/run_latent_distance.py <run_dir> --silhouette-sample 3000
  python scripts/run_latent_distance.py <run_dir> --scatter-patches 2000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
)

sys.path.insert(0, str(Path(__file__).parents[1]))
from subcellae.analysis.feature_analysis import latent_distance_histogram


# ── helpers ──────────────────────────────────────────────────────────────────

def _latent_cols(df: pd.DataFrame) -> list[str]:
    for prefix in ("lat_d", "z_"):
        cols = sorted([c for c in df.columns if c.startswith(prefix)],
                      key=lambda c: int(c.split(prefix)[1]))
        if cols:
            return cols
    raise ValueError("No latent columns found (expected lat_d* or z_*).")


def _load_cls_labels_all(variant_dir: Path, prefix: str) -> list[tuple[str, pd.DataFrame]]:
    """Return [(feature_set_name, label_df), ...] for every matching {prefix}_cls_* subdir.

    Handles both 'classification' and 'Position' ground-truth column names.
    """
    results = []
    for cls_dir in sorted(variant_dir.glob(f"{prefix}_cls_*")):
        pred_csv = cls_dir / "predictions_all.csv"
        if not pred_csv.exists():
            continue
        df = pd.read_csv(pred_csv)
        label_col = None
        for candidate in ("classification", "Position"):
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            continue
        labeled = df[df[label_col].notna() & (df[label_col] != "")].copy()
        if len(labeled) == 0:
            continue
        labeled = labeled.rename(columns={label_col: "classification"})
        labeled["filename"] = labeled["filename"].apply(lambda p: Path(p).name)
        results.append((cls_dir.name, labeled[["filename", "classification"]].copy()))
    return results


def _apply_exclude(labels, exclude):
    """Return boolean mask keeping samples whose label is not in exclude."""
    labels = np.asarray(labels, dtype=object)
    mask = np.ones(len(labels), dtype=bool)
    for v in exclude:
        mask &= labels != v
    return mask


def _run_hist(X, labels, save_path, metric, sample_pairs, title, exclude=()):
    mask = _apply_exclude(labels, exclude)
    if mask.sum() < 2:
        raise ValueError(f"Too few samples after filtering exclude={exclude}")
    return latent_distance_histogram(
        X[mask], np.asarray(labels, dtype=object)[mask],
        save_path=save_path,
        metric=metric,
        sample_pairs=sample_pairs,
        title=title,
    )


def _run_silhouette(X, labels, metric, exclude=(), sample_size=None, rng=None):
    """Silhouette score with optional random subsampling for speed."""
    labels = np.asarray(labels, dtype=object)
    mask = _apply_exclude(labels, exclude)
    # also drop NaN labels
    mask &= np.array([not (isinstance(v, float) and np.isnan(v)) for v in labels])
    X_f, y_f = X[mask], labels[mask]

    unique, counts = np.unique(y_f, return_counts=True)
    if len(unique) < 2:
        raise ValueError("Silhouette needs at least 2 classes.")
    if len(X_f) <= len(unique):
        raise ValueError("Silhouette needs more samples than classes.")

    # optional subsampling
    if sample_size is not None and len(X_f) > sample_size:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(X_f), size=sample_size, replace=False)
        X_f, y_f = X_f[idx], y_f[idx]
        unique, counts = np.unique(y_f, return_counts=True)
        if len(unique) < 2:
            raise ValueError("Silhouette subsampling left fewer than 2 classes.")

    singleton_classes = unique[counts < 2].tolist()
    global_score = float(silhouette_score(X_f, y_f, metric=metric))
    sample_scores = silhouette_samples(X_f, y_f, metric=metric)

    per_class = {}
    for cls in unique:
        s = sample_scores[y_f == cls]
        per_class[str(cls)] = {"mean": float(np.mean(s)), "n": int(len(s))}

    return {
        "silhouette_score": global_score,
        "silhouette_n_samples": int(len(X_f)),
        "silhouette_n_classes": int(len(unique)),
        "silhouette_singleton_classes": singleton_classes,
        "silhouette_per_class": per_class,
    }


def _run_ch(X, labels, exclude=()):
    """Calinski-Harabasz index."""
    labels = np.asarray(labels, dtype=object)
    mask = _apply_exclude(labels, exclude)
    X_f, y_f = X[mask], labels[mask]
    if len(np.unique(y_f)) < 2:
        raise ValueError("CH needs at least 2 classes.")
    return {"ch_score": float(calinski_harabasz_score(X_f, y_f))}


def _run_recon_scatter(variant_dir: Path, save_path: str,
                       max_patches: int = 1000, rng=None) -> dict:
    """Pixel-level input vs reconstruction hexbin scatter for one variant.

    Reads recon/patches_raw.tif and recon/patches_recon.tif (written by the
    analysis pipeline).  Each pixel becomes one point.  Subplots are split by
    condition.  Returns a dict with Pearson r values.
    """
    import tifffile
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    recon_dir = variant_dir / "recon"
    raw_tif = recon_dir / "patches_raw.tif"
    rec_tif = recon_dir / "patches_recon.tif"
    idx_csv = recon_dir / "patches_index.csv"

    if not raw_tif.exists() or not rec_tif.exists():
        raise FileNotFoundError(
            f"patches_raw.tif or patches_recon.tif missing in {recon_dir}")

    idx_df = pd.read_csv(idx_csv) if idx_csv.exists() else None

    raw_all = tifffile.imread(str(raw_tif))  # (N, [C,] H, W)
    rec_all = tifffile.imread(str(rec_tif))

    # collapse channel dim if present  →  (N, H, W)
    if raw_all.ndim == 4:
        raw_all = raw_all.reshape(raw_all.shape[0], -1,
                                  raw_all.shape[-2], raw_all.shape[-1]).mean(1)
        rec_all = rec_all.reshape(rec_all.shape[0], -1,
                                  rec_all.shape[-2], rec_all.shape[-1]).mean(1)

    N = raw_all.shape[0]
    rng = rng or np.random.default_rng(0)

    if max_patches and N > max_patches:
        sel = np.sort(rng.choice(N, size=max_patches, replace=False))
        raw_all = raw_all[sel]
        rec_all = rec_all[sel]
        if idx_df is not None:
            idx_df = idx_df.iloc[sel].reset_index(drop=True)

    # determine per-condition splits
    if idx_df is not None and "condition_name" in idx_df.columns:
        conditions = sorted(idx_df["condition_name"].unique().tolist())
        cond_masks = {c: (idx_df["condition_name"] == c).values for c in conditions}
    else:
        conditions = ["all"]
        cond_masks = {"all": np.ones(len(raw_all), dtype=bool)}

    n_cond = len(conditions)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 5))
    if n_cond == 1:
        axes = [axes]

    result = {"recon_n_patches": len(raw_all)}

    for ax, cond in zip(axes, conditions):
        m = cond_masks[cond]
        x = raw_all[m].ravel().astype(np.float64)
        y = rec_all[m].ravel().astype(np.float64)

        # Pearson r on a subsample for speed
        n_r = min(len(x), 200_000)
        ri = rng.choice(len(x), size=n_r, replace=False)
        r, _ = pearsonr(x[ri], y[ri])
        result[f"recon_pearson_r_{cond}"] = round(float(r), 6)

        hb = ax.hexbin(x, y, gridsize=80, cmap="viridis", mincnt=1, bins="log",
                       extent=[-0.2, 3, -0.2, 3])
        plt.colorbar(hb, ax=ax, label="log10(count)")

        ax.plot([-0.2, 3], [-0.2, 3], "r--", lw=1.2, label="y = x")
        ax.set_xlim(-0.2, 3)
        ax.set_ylim(-0.2, 3)
        ax.set_xlabel("Input pixel intensity")
        ax.set_ylabel("Reconstruction pixel intensity")
        ax.set_title(f"{cond}   r = {r:.4f}")
        ax.legend(loc="upper left", fontsize=8)

    # overall r across all patches
    x_all = raw_all.ravel().astype(np.float64)
    y_all = rec_all.ravel().astype(np.float64)
    n_r = min(len(x_all), 200_000)
    ri = rng.choice(len(x_all), size=n_r, replace=False)
    r_all, _ = pearsonr(x_all[ri], y_all[ri])
    result["recon_pearson_r"] = round(float(r_all), 6)

    fig.suptitle(
        f"{variant_dir.name} — input vs reconstruction (pixel level)\n"
        f"n_patches = {len(raw_all)}  |  overall r = {r_all:.4f}",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return result


# ── per-variant runner ────────────────────────────────────────────────────────

def _load_shared_labels(run_dir: Path) -> pd.DataFrame | None:
    """Scan all variant latents.csv files and return the first one that has
    annotation_label / annotation_label_2 populated.  Used to supply labels
    to variants whose own config didn't include an annotation file."""
    for variant_dir in sorted(run_dir.iterdir()):
        csv = variant_dir / "latents.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        if "annotation_label" not in df.columns:
            continue
        if (df["annotation_label"] != -1).any():
            df = df.copy()
            df["filename"] = df["filename"].apply(lambda p: Path(p).name)
            n_fa  = (df["annotation_label"]   != -1).sum()
            n_pos = (df.get("annotation_label_2", pd.Series([-1])) != -1).sum()
            print(f"  [shared labels] loaded from {variant_dir.name} "
                  f"(fa={n_fa}, pos={n_pos})")
            keep_cols = ["filename", "annotation_label", "annotation_label_name"]
            if "annotation_label_2" in df.columns:
                keep_cols += ["annotation_label_2", "annotation_label_2_name"]
            return df[keep_cols].copy()
    return None


def run_one(variant_dir: Path, metric: str, sample_pairs: int | None,
            silhouette_sample: int | None, scatter_patches: int = 1000,
            shared_labels: pd.DataFrame | None = None) -> list[dict]:
    csv = variant_dir / "latents.csv"
    if not csv.exists():
        print(f"  [skip] no latents.csv in {variant_dir.name}")
        return []

    df = pd.read_csv(csv)
    df = df.copy()
    df["filename"] = df["filename"].apply(lambda p: Path(p).name)
    latent_cols = _latent_cols(df)
    X = df[latent_cols].values
    rows = []

    def _try(label_type, Xf, labels, save_path, title, exclude=()):
        try:
            res = _run_hist(Xf, labels, save_path, metric, sample_pairs, title, exclude)
        except Exception as e:
            print(f"  {variant_dir.name:20s}  {label_type:30s}  SKIP (hist): {e}")
            return

        # silhouette
        sil_info = {}
        try:
            sil = _run_silhouette(Xf, labels, metric, exclude, silhouette_sample)
            res["silhouette_score"] = sil["silhouette_score"]
            sil_info = sil
        except Exception as e:
            print(f"    silhouette SKIP: {e}")

        # Calinski-Harabasz
        try:
            ch = _run_ch(Xf, labels, exclude)
            res.update(ch)
        except Exception as e:
            print(f"    CH SKIP: {e}")

        rows.append({"variant": variant_dir.name, "label_type": label_type, **res})

        ratio = res.get("mean_inter_over_intra_ratio", float("nan"))
        sil_v = res.get("silhouette_score", float("nan"))
        ch_v  = res.get("ch_score", float("nan"))
        print(f"  {variant_dir.name:20s}  {label_type:30s}  "
              f"ratio={ratio:.3f}  sil={sil_v:.3f}  ch={ch_v:.1f}")

        # print per-class silhouette breakdown
        if sil_info.get("silhouette_per_class"):
            for cls, d in sorted(sil_info["silhouette_per_class"].items()):
                print(f"      {cls:35s}  sil={d['mean']:.3f}  n={d['n']}")

    # ── condition ────────────────────────────────────────────────────────
    _try("condition", X,
         df["condition_name"].values,
         str(variant_dir / "latent_dist_condition.png"),
         f"{variant_dir.name} — condition (control vs ycomp)")

    # ── FA type (lat8 only) ───────────────────────────────────────────────
    fa_merged_for_combo = None
    for feat_set, fa_df in _load_cls_labels_all(variant_dir, "fa"):
        if "dist" in feat_set:
            continue
        merged = df[["filename"]].assign(_idx=range(len(df))).merge(
            fa_df.rename(columns={"classification": "fa_cls"}),
            on="filename", how="inner")
        print(f"  [debug] fa merge rows: {len(merged)}")
        _try("fa_type", X[merged["_idx"].values],
             merged["fa_cls"].values,
             str(variant_dir / "latent_dist_fa_type.png"),
             f"{variant_dir.name} — FA type (lat dims only)",
             exclude=("Uncertain",))
        fa_merged_for_combo = merged
        break

    # ── position (lat8 only) ──────────────────────────────────────────────
    _POS_EXCLUDE = ("Uncertain", "No Category/uncertain")
    pos_merged_for_combo = None
    for feat_set, pos_df in _load_cls_labels_all(variant_dir, "pos"):
        if "dist" in feat_set:
            continue
        merged = df[["filename"]].assign(_idx=range(len(df))).merge(
            pos_df.rename(columns={"classification": "pos_cls"}),
            on="filename", how="inner")
        print(f"  [debug] pos merge rows: {len(merged)}")
        _try("position", X[merged["_idx"].values],
             merged["pos_cls"].values,
             str(variant_dir / "latent_dist_position.png"),
             f"{variant_dir.name} — position (lat dims only)",
             exclude=_POS_EXCLUDE)
        pos_merged_for_combo = merged
        break

    # ── combined FA type × position ───────────────────────────────────────
    if fa_merged_for_combo is not None and pos_merged_for_combo is not None:
        combo = fa_merged_for_combo[["filename", "_idx", "fa_cls"]].merge(
            pos_merged_for_combo[["filename", "pos_cls"]],
            on="filename", how="inner")
        keep = (
            ~combo["fa_cls"].isin(("Uncertain",)) &
            ~combo["pos_cls"].isin(_POS_EXCLUDE)
        )
        combo = combo[keep].copy()
        print(f"  [debug] combo merge rows: {len(combo)}")
        if len(combo) >= 2:
            combo_labels = (combo["fa_cls"] + " | " + combo["pos_cls"]).values
            _try("fa_pos_combined",
                 X[combo["_idx"].values],
                 combo_labels,
                 str(variant_dir / "latent_dist_fa_pos_combined.png"),
                 f"{variant_dir.name} — FA type × position")

    # ── FA type / position from annotation ───────────────────────────────
    # Use this variant's own latents.csv if annotated; otherwise fall back
    # to shared_labels loaded from another variant in the same run.
    ann_df = df  # candidate source of annotation columns
    if not ("annotation_label" in df.columns and (df["annotation_label"] != -1).any()):
        if shared_labels is not None:
            # merge shared labels into this variant's row order by filename
            ann_df = df[["filename"]].assign(_idx=range(len(df))).merge(
                shared_labels, on="filename", how="left")
            ann_df = ann_df.set_index("_idx").reindex(range(len(df)))
            ann_df.index = df.index
        else:
            ann_df = None

    has_fa_ann  = ann_df is not None and \
                  "annotation_label" in ann_df.columns and \
                  (ann_df["annotation_label"].fillna(-1) != -1).any()
    has_pos_ann = ann_df is not None and \
                  "annotation_label_2" in ann_df.columns and \
                  (ann_df["annotation_label_2"].fillna(-1) != -1).any()

    if has_fa_ann and fa_merged_for_combo is None:
        mask_fa = (ann_df["annotation_label"].fillna(-1) != -1).values
        _try("fa_type",
             X[mask_fa],
             ann_df.loc[mask_fa, "annotation_label_name"].values,
             str(variant_dir / "latent_dist_fa_type.png"),
             f"{variant_dir.name} — FA type (annotation)",
             exclude=("Uncertain",))

    if has_pos_ann and pos_merged_for_combo is None:
        mask_pos = (ann_df["annotation_label_2"].fillna(-1) != -1).values
        _try("position",
             X[mask_pos],
             ann_df.loc[mask_pos, "annotation_label_2_name"].values,
             str(variant_dir / "latent_dist_position.png"),
             f"{variant_dir.name} — position (annotation)",
             exclude=_POS_EXCLUDE)

    # ── combined FA × position from annotation ────────────────────────────
    if fa_merged_for_combo is None and pos_merged_for_combo is None:
        if has_fa_ann and has_pos_ann:
            mask_both = (
                (ann_df["annotation_label"].fillna(-1)   != -1) &
                (ann_df["annotation_label_2"].fillna(-1) != -1)
            ).values
            fa_labels  = ann_df.loc[mask_both, "annotation_label_name"].values
            pos_labels = ann_df.loc[mask_both, "annotation_label_2_name"].values
            keep = (
                ~np.isin(fa_labels,  ["Uncertain"]) &
                ~np.isin(pos_labels, list(_POS_EXCLUDE))
            )
            if keep.sum() >= 2:
                combo_labels = np.array(
                    [f"{f} | {p}" for f, p in zip(fa_labels[keep], pos_labels[keep])])
                idx_both = np.where(mask_both)[0][keep]
                _try("fa_pos_combined",
                     X[idx_both],
                     combo_labels,
                     str(variant_dir / "latent_dist_fa_pos_combined.png"),
                     f"{variant_dir.name} — FA type × position (annotation)")

    # ── reconstruction pixel scatter ─────────────────────────────────────
    try:
        res = _run_recon_scatter(
            variant_dir,
            str(variant_dir / "recon_scatter.png"),
            max_patches=scatter_patches,
        )
        rows.append({"variant": variant_dir.name, "label_type": "recon_pixels", **res})
        print(f"  {variant_dir.name:20s}  {'recon_pixels':30s}  "
              f"Pearson r={res.get('recon_pearson_r', float('nan')):.4f}  "
              f"n_patches={res['recon_n_patches']}")
    except Exception as e:
        print(f"  {variant_dir.name:20s}  {'recon_pixels':30s}  SKIP: {e}")

    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--sample-pairs", type=int, default=50_000,
                        help="pairs sampled for distance histogram (default 50000)")
    parser.add_argument("--silhouette-sample", type=int, default=5_000,
                        help="max samples for silhouette (0 = use all, default 5000)")
    parser.add_argument("--scatter-patches", type=int, default=1_000,
                        help="patches sampled for recon scatter (0 = use all, default 1000)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    sil_sample     = args.silhouette_sample if args.silhouette_sample > 0 else None
    scatter_patches = args.scatter_patches  if args.scatter_patches   > 0 else None

    print(f"Run dir          : {run_dir}")
    print(f"Metric           : {args.metric}")
    print(f"Pairs            : {args.sample_pairs}")
    print(f"Silhouette sample: {sil_sample or 'all'}")
    print(f"Scatter patches  : {scatter_patches or 'all'}")
    print()

    shared_labels = _load_shared_labels(run_dir)

    all_rows = []
    for variant_dir in sorted(run_dir.iterdir()):
        if variant_dir.is_dir():
            all_rows.extend(run_one(variant_dir, args.metric,
                                    args.sample_pairs, sil_sample,
                                    scatter_patches=scatter_patches or 0,
                                    shared_labels=shared_labels))

    if all_rows:
        summary = pd.DataFrame(all_rows)
        out_csv = run_dir / "latent_distance_summary.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary → {out_csv}")
        show_cols = [c for c in ["variant", "label_type",
                                 "mean_intra", "mean_inter",
                                 "mean_inter_over_intra_ratio",
                                 "silhouette_score", "ch_score",
                                 "recon_pearson_r", "recon_n_patches"]
                     if c in summary.columns]
        print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
