import os
import pandas as pd
from helpers import ctrl_Y_ID_adding, LABEL_COLS

# ── CONFIG ────────────────────────────────────────────────────────────────────
LABEL_DIR  = '/mnt/d/lding/FA/analysis_results/labelling'
BATCH1_CSV = os.path.join(LABEL_DIR, 'paxdata_paxpatch_batch1_labels.csv')
OUT_PATH   = os.path.join(LABEL_DIR, 'paxdata_paxpatch_batch1and2_combined_labels.csv')

# ── LOAD ──────────────────────────────────────────────────────────────────────
batch1      = pd.read_csv(BATCH1_CSV)
pi_Y_df     = ctrl_Y_ID_adding(pd.read_csv(os.path.join(LABEL_DIR, 'Margaret_Ycomp_V2_project-20-at-2026-02-09-21-40-9d1e4d7c.csv')))
pi_C_df     = ctrl_Y_ID_adding(pd.read_csv(os.path.join(LABEL_DIR, 'Margaret_Control_V2_project-19-at-2026-02-09-21-46-5552392a.csv')))

# ── STEP 1: update batch1 with new control labels (pi_C wins on duplicates) ──
batch1_updated = (
    pd.concat([batch1, pi_C_df[LABEL_COLS]], ignore_index=True)
      .drop_duplicates(subset="unique_ID", keep="last")
)

# ── STEP 2: update batch1 with pi_Y labels where they overlap ────────────────
dup_ids = batch1_updated["unique_ID"].isin(pi_Y_df["unique_ID"])
dup_id_values = batch1_updated.loc[dup_ids, "unique_ID"].unique()
print(f"Overlapping IDs between batch1 and pi_Y: {len(dup_id_values)}")

overwrite_cols = [c for c in LABEL_COLS if c != "unique_ID"]
b1_idx = batch1_updated.set_index("unique_ID")
pi_idx = pi_Y_df.set_index("unique_ID")
b1_idx.loc[dup_id_values, overwrite_cols] = pi_idx.loc[dup_id_values, overwrite_cols]
batch1_updated = b1_idx.reset_index()

# ── STEP 3: append remaining pi_Y rows not in batch1 ─────────────────────────
pi_Y_remaining = pi_Y_df[~pi_Y_df["unique_ID"].isin(dup_id_values)]
print(f"New pi_Y rows added: {len(pi_Y_remaining)}")

# ── COMBINE & EXPORT ──────────────────────────────────────────────────────────
combined = (
    pd.concat([batch1_updated, pi_Y_remaining[LABEL_COLS]], ignore_index=True)
      .drop_duplicates(subset="unique_ID", keep="last")
)

combined[LABEL_COLS].to_csv(OUT_PATH, index=False)
print(f"Saved {len(combined)} labels → {OUT_PATH}")
