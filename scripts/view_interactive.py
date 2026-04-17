"""
Interactive Patch Viewer — Panel + Bokeh

Two-direction exploration:

  Direction A — UMAP → detail
    Hover a UMAP dot  →  raw + recon patch appear in tooltip (instant, client-side).
    Tap a UMAP dot    →  detail panel below updates (patch images + prediction text).

  Direction B — Image → UMAP
    Choose an image from the dropdown (condition × source image).
    The full paxillin canvas is shown with coloured patch rectangles.
    Click anywhere on the canvas  →  the nearest patch is found, its UMAP point is
    highlighted with a large red dot, and the prediction text updates below.

Layout
------
  ┌──────────────────────┬────────────────────────────┐
  │  [Color ▼]           │  [Image selector ▼]        │
  │  UMAP scatter        │  Full paxillin canvas       │
  │  (hover = tooltip)   │  (click = UMAP highlight)   │
  │  (tap   = detail ↓)  │  (colored patch boxes)      │
  ├──────────────────────┴────────────────────────────┤
  │  FA: …  Position: …   |  Raw patch  |  Recon patch │
  └─────────────────────────────────────────────────────┘

Usage
-----
    python scripts/view_interactive.py path/to/interactive.h5
    panel serve scripts/view_interactive.py --args path/to/interactive.h5 --show
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import h5py
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import Tap
from bokeh.layouts import column as bk_column
from bokeh.models import (
    ColumnDataSource, CustomJS, HoverTool,
    LinearColorMapper, Range1d, Select,
)
from bokeh.plotting import figure

pn.extension(sizing_mode='stretch_width')

# ── Shared colour palettes ────────────────────────────────────────────────────
FA_ORDER   = ["Nascent Adhesion", "focal complex", "focal adhesion",
              "fibrillar adhesion", "No adhesion"]
POS_ORDER  = ["Cell Protruding Edge", "Cell Periphery/other", "Lamella", "Cell Body"]
FA_COLORS  = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#aaaaaa"]
POS_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8"]
FALLBACK   = "#cccccc"

# Grayscale palette: index 0 → black, index 255 → white
try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]


def _label_color(label: str, order: list, colors: list) -> str:
    try:
        return colors[order.index(str(label))]
    except (ValueError, IndexError):
        return FALLBACK


# ── HDF5 loading ──────────────────────────────────────────────────────────────

def load_h5(path: str):
    """Return (df, patches_raw, patches_recon, images_raw, img_meta, pad, scale, result_dir)."""
    with h5py.File(path, 'r') as f:
        df            = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        patches_raw   = f['patches/raw'][()]   if 'patches/raw'   in f else None
        patches_recon = f['patches/recon'][()] if 'patches/recon' in f else None
        images_raw    = f['images/raw'][()]    if 'images/raw'    in f else None
        img_meta      = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                         if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
        result_dir  = Path(str(f.attrs.get('result_dir', '')))
    return df, patches_raw, patches_recon, images_raw, img_meta, pad_size, image_scale, result_dir


# ── Image helpers ─────────────────────────────────────────────────────────────

def _norm_image(arr: np.ndarray) -> np.ndarray:
    """Normalise to float32 [0, 1]."""
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _flip_for_bokeh(arr: np.ndarray) -> np.ndarray:
    """Flip vertically so array row-0 renders at the top of a Bokeh figure.

    Bokeh's 'image' renderer places array row-0 at y=0 (bottom).  Flipping
    gives the correct top-down orientation for microscopy images.

    Coordinate mapping after flipud:
        Bokeh y-coordinate  ↔  original array row (H - 1 - bokeh_y ≈ array_row)
        For a tap at bokeh_y:  array_row ≈ H - bokeh_y
    """
    return np.ascontiguousarray(np.flipud(arr))


def _get_frame(images_raw: np.ndarray, img_meta: pd.DataFrame,
               group: str, channel: int = 0) -> np.ndarray | None:
    """Return the canvas array for a given (group, channel)."""
    matches = img_meta[img_meta['group'].astype(str) == group]
    if matches.empty:
        return None
    if 'channel' in matches.columns:
        ch0 = matches[matches['channel'] == channel]
        row = ch0.iloc[0] if not ch0.empty else matches.iloc[0]
    else:
        row = matches.iloc[0]
    return images_raw[int(row['frame'])]


# ── Matplotlib figure helpers ─────────────────────────────────────────────────

def _fig_to_pane(fig: plt.Figure, dpi: int = 130) -> pn.pane.PNG:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return pn.pane.PNG(buf, sizing_mode='scale_width')


def _patch_figure(raw: np.ndarray, recon: np.ndarray, title: str = '') -> pn.pane.PNG:
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))
    for ax, arr, lbl in zip(axes, [raw, recon], ['Raw', 'Recon']):
        ax.imshow(arr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(lbl, fontsize=10)
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=8, y=1.01)
    fig.tight_layout(pad=0.3)
    return _fig_to_pane(fig)


# ── Legend HTML ───────────────────────────────────────────────────────────────

def _legend_html() -> str:
    rows = ['<div style="font-size:11px;line-height:1.8;"><b>FA type</b>']
    for lbl, col in zip(FA_ORDER, FA_COLORS):
        rows.append(
            f'<span style="background:{col};display:inline-block;'
            f'width:11px;height:11px;margin-right:5px;border-radius:2px;'
            f'vertical-align:middle;"></span>{lbl}'
        )
    rows.append('<br><b>Position</b>')
    for lbl, col in zip(POS_ORDER, POS_COLORS):
        rows.append(
            f'<span style="background:{col};display:inline-block;'
            f'width:11px;height:11px;margin-right:5px;border-radius:2px;'
            f'vertical-align:middle;"></span>{lbl}'
        )
    rows.append('</div>')
    return '<br>'.join(rows)


# ── Main app ──────────────────────────────────────────────────────────────────

def build_app(h5_path: str) -> pn.viewable.Viewable:
    print(f'[view] Loading {h5_path} ...')
    (df, patches_raw, patches_recon,
     images_raw, img_meta, pad_size, image_scale, result_dir) = load_h5(h5_path)
    n = len(df)
    print(f'[view]   {n} patches, image_scale={image_scale}')

    # ── Old-format fallback: individual TIFF files on disk ────────────────────
    recon_patches_dir = result_dir / 'recon' / 'patches'
    recon_images_dir  = result_dir / 'recon' / 'images'
    has_old_patches = (patches_raw is None and result_dir != Path('')
                       and recon_patches_dir.is_dir())
    old_img_files: list = []
    has_old_images = False
    if images_raw is None and img_meta is None and result_dir != Path(''):
        old_img_files = sorted(recon_images_dir.glob('raw_*.tif'))
        has_old_images = len(old_img_files) > 0
    if has_old_patches:
        print(f'[view]   Old-format patches found in {recon_patches_dir}')
    if has_old_images:
        print(f'[view]   Old-format images: {len(old_img_files)} files')

    # Fall back to latent dims if UMAP not available
    if 'UMAP_1' not in df.columns:
        z_cols = [c for c in df.columns if c.startswith('z_')]
        if len(z_cols) >= 2:
            df['UMAP_1'], df['UMAP_2'] = df[z_cols[0]], df[z_cols[1]]
            print(f'[view]   No UMAP -- showing {z_cols[0]} vs {z_cols[1]}')
        else:
            df['UMAP_1'] = df['UMAP_2'] = 0.0

    fa_pred  = df.get('fa_pred',  pd.Series([''] * n)).fillna('').astype(str)
    pos_pred = df.get('pos_pred', pd.Series([''] * n)).fillna('').astype(str)

    # ── UMAP ColumnDataSource ─────────────────────────────────────────────────
    umap_data: dict = dict(
        x         = df['UMAP_1'].fillna(0).values,
        y         = df['UMAP_2'].fillna(0).values,
        idx       = np.arange(n, dtype=int),
        condition = (df.get('condition_name', df.get('condition', pd.Series([''] * n)))
                     .fillna('').astype(str).values),
        fa_pred   = fa_pred.values,
        pos_pred  = pos_pred.values,
        filename  = df['filename'].astype(str).values,
        color_fa  = [_label_color(v, FA_ORDER,  FA_COLORS)  for v in fa_pred],
        color_pos = [_label_color(v, POS_ORDER, POS_COLORS) for v in pos_pred],
    )
    umap_data['color'] = list(umap_data['color_fa'])
    has_b64 = 'raw_b64' in df.columns
    if has_b64:
        umap_data['raw_b64']   = df['raw_b64'].values
        umap_data['recon_b64'] = df['recon_b64'].values

    umap_src = ColumnDataSource(umap_data)

    # Single big red dot on UMAP -- updated when user clicks the image panel
    highlight_src = ColumnDataSource({'x': [], 'y': []})

    # ── UMAP scatter figure ───────────────────────────────────────────────────
    p_umap = figure(
        width=520, height=500,
        title='UMAP  (hover = patch tooltip  |  tap = detail panel)',
        tools='pan,wheel_zoom,box_zoom,reset,tap',
        toolbar_location='above',
    )

    scatter = p_umap.scatter(
        'x', 'y', source=umap_src, marker='circle',
        fill_color='color', line_color='color',
        size=5, alpha=0.65,
        nonselection_fill_color='color', nonselection_fill_alpha=0.15,
        nonselection_line_alpha=0.0,
        selection_fill_color='color', selection_line_color='white',
        selection_line_width=1.5,
    )

    # Highlighted point (from image click) -- drawn on top as a large red dot
    p_umap.scatter(
        'x', 'y', source=highlight_src, marker='circle',
        fill_color='red', line_color='white',
        size=18, alpha=1.0, line_width=2.5,
    )

    # Hover tooltip with embedded patch images (client-side, instant)
    if has_b64:
        hover_html = """
            <div style="background:#111;padding:6px 8px;border-radius:6px;max-width:310px;">
              <div style="display:flex;gap:6px;">
                <div style="text-align:center;">
                  <img src="data:image/png;base64,@raw_b64"
                       style="width:128px;height:128px;image-rendering:pixelated;display:block;"/>
                  <span style="color:#aaa;font-size:10px;">Raw</span>
                </div>
                <div style="text-align:center;">
                  <img src="data:image/png;base64,@recon_b64"
                       style="width:128px;height:128px;image-rendering:pixelated;display:block;"/>
                  <span style="color:#aaa;font-size:10px;">Recon</span>
                </div>
              </div>
              <div style="color:#ccc;font-size:10px;margin-top:5px;line-height:1.4;">
                @filename<br>cond: @condition<br>FA: @fa_pred<br>Pos: @pos_pred
              </div>
            </div>"""
        p_umap.add_tools(HoverTool(renderers=[scatter], tooltips=hover_html))
    else:
        p_umap.add_tools(HoverTool(renderers=[scatter], tooltips=[
            ('file', '@filename'), ('cond', '@condition'),
            ('FA',   '@fa_pred'),  ('Pos',  '@pos_pred'),
        ]))

    p_umap.xaxis.axis_label = 'UMAP 1'
    p_umap.yaxis.axis_label = 'UMAP 2'

    # Colour-by selector (pure JS -- no server round-trip)
    color_select = Select(
        title='Colour by', value='fa_pred',
        options=[('fa_pred', 'FA type'), ('pos_pred', 'Position')],
        width=180,
    )
    color_select.js_on_change('value', CustomJS(
        args=dict(src=umap_src, plot=p_umap), code="""
        const d = src.data;
        d['color'] = (cb_obj.value === 'fa_pred')
            ? [...d['color_fa']] : [...d['color_pos']];
        src.change.emit();
        const lbl = (cb_obj.value === 'fa_pred') ? 'FA type' : 'Position';
        plot.title.text = 'UMAP  -- ' + lbl
            + '  (hover = patch tooltip  |  tap = detail panel)';
    """))

    left_col = pn.pane.Bokeh(bk_column(color_select, p_umap))

    # ── Full image Bokeh figure (Direction B) ─────────────────────────────────
    has_images = (images_raw is not None and img_meta is not None) or has_old_images

    # Placeholders updated inside the has_images block
    rects_src = sel_src = img_fig = img_pane = img_select_widget = None
    _state: dict = {}

    if has_images:
        # Build selector options: "condition | group_key"
        pg_col   = 'patch_group' if 'patch_group' in df.columns else 'group'
        cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'
        grp_to_cond: dict = {}
        for _, row in df[[pg_col, cond_col]].dropna().drop_duplicates().iterrows():
            grp_to_cond[str(row[pg_col])] = str(row[cond_col])

        if images_raw is not None and img_meta is not None:
            # Packed format: images stored in HDF5 array
            unique_groups = sorted(img_meta['group'].astype(str).unique())
            def _get_canvas(group_key: str) -> np.ndarray:
                return _norm_image(_get_frame(images_raw, img_meta, group_key))
        else:
            # Old-format: individual TIFFs on disk (raw_{group_key}.tif)
            unique_groups = sorted(p.stem[4:] for p in old_img_files)  # strip 'raw_'
            def _get_canvas(group_key: str) -> np.ndarray:
                p = recon_images_dir / f'raw_{group_key}.tif'
                arr = tifffile.imread(str(p)).astype(np.float32)
                if arr.ndim == 3:
                    arr = arr[0]
                mx = arr.max()
                return arr / mx if mx > 0 else arr

        img_options = {f"{grp_to_cond.get(g, '?')} | {g}": g
                       for g in unique_groups}

        init_group = unique_groups[0]
        init_arr   = _get_canvas(init_group)
        H, W       = init_arr.shape[:2]

        # Image data source
        img_src = ColumnDataSource(dict(
            image=[_flip_for_bokeh(init_arr)],
            x=[0], y=[0], dw=[W], dh=[H],
        ))

        # Patch rectangle source (coloured by FA prediction)
        rects_src = ColumnDataSource(dict(
            x=[], y=[], width=[], height=[], color=[], df_idx=[],
        ))
        # Selected-patch white-border highlight
        sel_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

        # Figure
        img_fig = figure(
            width=520, height=520,
            x_range=Range1d(0, W),
            y_range=Range1d(0, H),
            title='Full paxillin canvas  (click a patch to highlight on UMAP)',
            tools='tap,pan,wheel_zoom,reset',
            toolbar_location='above',
        )
        gray_mapper = LinearColorMapper(palette=GRAY256, low=0.0, high=1.0)
        img_fig.image(
            image='image', source=img_src,
            x=0, y=0, dw=W, dh=H,
            color_mapper=gray_mapper,
        )
        img_fig.rect(
            'x', 'y', 'width', 'height', source=rects_src,
            fill_alpha=0, line_color='color', line_width=0.9, line_alpha=0.75,
        )
        img_fig.rect(
            'x', 'y', 'width', 'height', source=sel_src,
            fill_alpha=0, line_color='white', line_width=2.5,
        )
        img_fig.xaxis.axis_label = 'column (px)'
        img_fig.yaxis.axis_label = 'row (px)'

        # Helper: build rect data for a group (in Bokeh flipped-y coordinates)
        def _rects_for_group(group_key: str, img_H: int) -> dict:
            mask = df[pg_col].astype(str) == group_key
            sub  = df[mask]
            xs, ys, ws, hs, cols, idxs = [], [], [], [], [], []
            for i, row in sub.iterrows():
                cx = row.get('canvas_cx', np.nan)
                cy = row.get('canvas_cy', np.nan)
                ps = int(row.get('ps', 32))
                if pd.isna(cx) or pd.isna(cy):
                    continue
                # With flipud display: Bokeh_y = img_H - canvas_cy
                xs.append(float(cx) * image_scale)
                ys.append((img_H - float(cy)) * image_scale)
                ws.append(float(ps) * image_scale)
                hs.append(float(ps) * image_scale)
                cols.append(_label_color(
                    str(row.get('fa_pred', '')), FA_ORDER, FA_COLORS))
                idxs.append(i)
            return dict(x=xs, y=ys, width=ws, height=hs, color=cols, df_idx=idxs)

        _state.update(group=init_group, H=H, W=W)
        rects_src.data = _rects_for_group(init_group, H)

        def _load_group(group_key: str) -> None:
            arr     = _get_canvas(group_key)
            Hn, Wn  = arr.shape[:2]
            img_src.data = dict(
                image=[_flip_for_bokeh(arr)],
                x=[0], y=[0], dw=[Wn], dh=[Hn],
            )
            img_fig.x_range.start, img_fig.x_range.end = 0, Wn
            img_fig.y_range.start, img_fig.y_range.end = 0, Hn
            rects_src.data = _rects_for_group(group_key, Hn)
            sel_src.data   = dict(x=[], y=[], width=[], height=[])
            highlight_src.data = dict(x=[], y=[])
            _state.update(group=group_key, H=Hn, W=Wn)

        img_select_widget = pn.widgets.Select(
            name='Image', options=img_options,
            value=init_group, width=420,
        )
        img_select_widget.param.watch(lambda e: _load_group(e.new), 'value')
        img_pane = pn.pane.Bokeh(img_fig)

    # ── Shared detail panel (bottom bar) ──────────────────────────────────────
    pred_md   = pn.pane.Markdown(
        '*Hover the UMAP for a quick patch preview.  '
        'Tap the UMAP **or** click a patch in the canvas for full details.*',
        width=520,
    )
    patch_col = pn.Column(pn.pane.Markdown(''), width=480)

    def _show_detail(idx: int) -> None:
        row   = df.iloc[idx]
        fa    = str(row.get('fa_pred',  '--'))
        pos   = str(row.get('pos_pred', '--'))
        fname = str(row.get('filename', ''))
        cond  = str(row.get('condition_name', row.get('condition', '')))
        pred_md.object = (
            f"**Patch:** `{Path(fname).stem}`  ·  **Condition:** {cond}  \n"
            f"**FA type:** {fa}  ·  **Position:** {pos}"
        )
        if patches_raw is not None:
            patch_col.objects = [_patch_figure(
                patches_raw[idx], patches_recon[idx],
                title=Path(fname).stem,
            )]
        elif has_old_patches:
            stem    = Path(fname).stem
            raw_p   = recon_patches_dir / f'raw_{stem}.tif'
            recon_p = recon_patches_dir / f'recon_{stem}.tif'
            if raw_p.exists() and recon_p.exists():
                raw_arr   = tifffile.imread(str(raw_p)).astype(np.float32)
                recon_arr = tifffile.imread(str(recon_p)).astype(np.float32)
                if raw_arr.ndim == 3:
                    raw_arr = raw_arr[0]
                if recon_arr.ndim == 3:
                    recon_arr = recon_arr[0]
                for _a in [raw_arr, recon_arr]:
                    mx = _a.max()
                    if mx > 0:
                        _a /= mx
                patch_col.objects = [_patch_figure(raw_arr, recon_arr, title=stem)]
            else:
                patch_col.objects = [pn.pane.Markdown(f'*Patch files not found for {stem}*')]

    # ── Direction A: UMAP tap → detail + canvas highlight ─────────────────────
    def _on_umap_tap(attr, old, new):
        if not new:
            return
        idx = int(new[0])
        _show_detail(idx)

        if not has_images:
            return
        row    = df.iloc[idx]
        pg_val = str(row.get(pg_col, ''))
        cx     = row.get('canvas_cx', np.nan)
        cy     = row.get('canvas_cy', np.nan)
        ps     = float(row.get('ps', 32))
        H_cur  = _state['H']
        if not pd.isna(cx) and not pd.isna(cy):
            bx = float(cx) * image_scale
            by = (H_cur - float(cy)) * image_scale
            sel_src.data = dict(
                x=[bx], y=[by],
                width=[ps * image_scale], height=[ps * image_scale],
            )
            # Switch image panel to the source image containing this patch
            if pg_val and pg_val != _state['group']:
                _load_group(pg_val)
                img_select_widget.value = pg_val

    umap_src.selected.on_change('indices', _on_umap_tap)

    # ── Direction B: image click → UMAP highlight + detail ───────────────────
    if has_images:
        def _on_image_tap(event: Tap) -> None:
            H_cur       = _state['H']
            # Convert Bokeh click coordinates back to canvas (original array) space
            canvas_cx_t = event.x / image_scale
            canvas_cy_t = (H_cur - event.y) / image_scale   # undo flipud

            # Patch centres in canvas coordinates
            xs_bk  = np.array(rects_src.data['x'],      dtype=float)
            ys_bk  = np.array(rects_src.data['y'],      dtype=float)
            df_idx = np.array(rects_src.data['df_idx'], dtype=int)

            if len(xs_bk) == 0:
                return

            # Convert rect centres back to canvas coords for distance calculation
            cx_arr = xs_bk / image_scale
            cy_arr = (H_cur - ys_bk) / image_scale

            dists   = np.sqrt((cx_arr - canvas_cx_t)**2 + (cy_arr - canvas_cy_t)**2)
            near_i  = int(np.argmin(dists))
            near_df = int(df_idx[near_i])

            # Big red dot on UMAP
            row    = df.iloc[near_df]
            umap_x = float(row.get('UMAP_1', row.get('umap_1', 0)) or 0)
            umap_y = float(row.get('UMAP_2', row.get('umap_2', 0)) or 0)
            highlight_src.data = dict(x=[umap_x], y=[umap_y])

            # White-border highlight on canvas
            sel_src.data = dict(
                x=[xs_bk[near_i]],
                y=[ys_bk[near_i]],
                width=[rects_src.data['width'][near_i]],
                height=[rects_src.data['height'][near_i]],
            )

            _show_detail(near_df)

        img_fig.on_event(Tap, _on_image_tap)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = pn.pane.HTML(_legend_html(), width=480)

    # ── Layout assembly ───────────────────────────────────────────────────────
    bottom_bar = pn.Column(
        pn.layout.Divider(),
        pn.pane.Markdown('### Selected patch details', width=1060),
        pred_md,
        pn.Row(patch_col, legend),
    )

    if has_images:
        right_col = pn.Column(img_select_widget, img_pane, width=540)
    else:
        right_col = pn.pane.Markdown(
            '*Full image data not in this HDF5.*\n\n'
            'Re-pack with `--image-scale 1.0` (default) to include canvas images.',
            width=540,
        )

    return pn.Column(
        pn.pane.Markdown(
            f'## Interactive Patch Viewer  ·  `{Path(h5_path).name}`',
            sizing_mode='stretch_width',
        ),
        pn.Row(left_col, pn.Spacer(width=12), right_col,
               sizing_mode='stretch_width'),
        bottom_bar,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _get_h5_path() -> str:
    sess = pn.state.session_args
    if 'args' in sess and sess['args']:
        arg = sess['args'][0]
        return arg.decode() if isinstance(arg, bytes) else str(arg)
    if len(sys.argv) > 1:
        return sys.argv[1]
    print('Usage: python scripts/view_interactive.py path/to/interactive.h5',
          file=sys.stderr)
    sys.exit(1)


if pn.state.served:
    build_app(_get_h5_path()).servable()

if __name__ == '__main__':
    h5_path = _get_h5_path()
    # Serve as a single app at '/' so the browser opens the correct URL.
    # (Using a named dict routes to '/Name%20With%20Space' which gives 404.)
    pn.serve(build_app(h5_path), show=True, port=5006, autoreload=False)
