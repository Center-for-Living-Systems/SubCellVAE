---
marp: true
theme: default
paginate: false
style: |
  section {
    font-size: 21px;
    font-family: Arial, sans-serif;
  }
  h1 { font-size: 34px; color: #1a4a7a; }
  h2 { font-size: 26px; color: #1a4a7a; border-bottom: 2px solid #1a4a7a; padding-bottom: 4px; }
  table { font-size: 18px; border-collapse: collapse; width: 100%; }
  th { background: #1a4a7a; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; border: 1px solid #ccc; }
  tr:nth-child(even) { background: #f2f6fb; }
  .note  { background: #fff8e1; border-left: 4px solid #f0a500; padding: 8px 14px; margin-top: 10px; }
  .columns { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 2em; }
  .small { font-size: 17px; color: #555; }
---

## Distance-to-Edge Feature Vector (`dist8`)

<div class="columns">
<div>

**What it encodes:**
Each patch center casts **8 rays** at evenly-spaced angles (0°, 45°, 90°, …, 315°) outward through the cell mask, recording the pixel distance to the nearest cell boundary in each direction.

**Rotation invariance:**
The 8 distances are **cyclically shifted** so that the direction of *minimum* distance is always placed first (`d00`). This makes the feature independent of cell orientation.

**Result:** an 8-dimensional vector `[d00, d01, …, d07]` where:
- `d00` = shortest ray (closest edge)
- `d04` ≈ ray roughly opposite to `d00`

**Used as:** supplement to the AE latent `z` for LightGBM classification (`lat8 + dist8` or `lat16 + dist8`)

</div>
<div>

```
         d02  d01
          |  /
    d03 --•-- d00  ← min distance (always d00)
          |  \
         d04  d07
           ↓
     cell boundary
```

<br>

| Feature | Meaning |
|---------|---------|
| `d00` | distance to nearest edge |
| `d01`–`d03` | distances at +45°, +90°, +135° |
| `d04` | distance to opposite side |
| `d05`–`d07` | remaining directions |

<div class="note small">

**Why it helps:** subcellular position (edge vs. lamella vs. cell body) is strongly encoded in the relative distances — a patch at the cell periphery has one very short ray; a central patch has long rays in all directions.

</div>

</div>
</div>
