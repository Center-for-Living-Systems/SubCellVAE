import os
import pandas as pd
from helpers import ctrl_Y_ID_adding, LABEL_COLS

# ── CONFIG ────────────────────────────────────────────────────────────────────
LABEL_DIR = '/mnt/d/lding/FA/analysis_results/labelling'
OUT_PATH  = '/mnt/d/lding/FA/analysis_results/labelling/paxdata_paxpatch_batch1_labels.csv'

# ── LOAD ──────────────────────────────────────────────────────────────────────
df_C_day1218 = ctrl_Y_ID_adding(pd.read_csv(os.path.join(LABEL_DIR, 'project-13-at-2025-12-18-15-32-df671bd2.csv')))
df_C_day1219 = ctrl_Y_ID_adding(pd.read_csv(os.path.join(LABEL_DIR, 'project-13-at-2025-12-19-15-41-effff775.csv')))
df_Y_day1222 = ctrl_Y_ID_adding(pd.read_csv(os.path.join(LABEL_DIR, 'project-15-at-2025-12-22-18-44-b7e23381.csv')))

# ── MERGE ─────────────────────────────────────────────────────────────────────
# Two control sessions: keep df_C_day1219 labels on duplicates (more recent)
df_C_merged = (
    pd.concat([df_C_day1218, df_C_day1219], ignore_index=True)
      .drop_duplicates(subset="crop_img_filename", keep="last")
)

df_all = pd.concat([df_C_merged, df_Y_day1222], ignore_index=True)

# ── EXPORT ────────────────────────────────────────────────────────────────────
df_all[LABEL_COLS].to_csv(OUT_PATH, index=False)
print(f"Saved {len(df_all)} labels → {OUT_PATH}")
