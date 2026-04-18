"""
label_patches.py
================
Manual patch labelling tool.  Reads the same interactive.h5 produced by
pack_interactive_h5.py.  Shows one full canvas image at a time with all patch
boxes drawn as a grid.  Click a label button to set the active label, then
click any patch box on the canvas to assign that label to the patch.

Labels are stored as { filename → label } and written to a CSV on "Finish".

Usage
-----
    python scripts/label_patches.py path/to/interactive.h5
    python scripts/label_patches.py path/to/interactive.h5 --out my_labels.csv
    python scripts/label_patches.py path/to/interactive.h5 --port 5007
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import panel as pn
import tifffile
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, LinearColorMapper, Range1d
from bokeh.plotting import figure

from subcellae.utils.label_colors import (
    classification_label_to_color as FA_COLOR_MAP,
)

pn.extension(sizing_mode='stretch_width')

# Labels available in this tool
LABEL_OPTIONS = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "No adhesion",
]
UNLABELED_COLOR = "#555555"

try:
    from bokeh.palettes import gray as _bk_gray
    GRAY256 = _bk_gray(256)
except Exception:
    GRAY256 = [f'#{i:02x}{i:02x}{i:02x}' for i in range(256)]


# ── HDF5 loading ──────────────────────────────────────────────────────────────

def load_h5(path: str):
    with h5py.File(path, 'r') as f:
        df         = pd.read_csv(io.StringIO(f['meta/csv'][()].decode()))
        images_raw = f['images/raw'][()]    if 'images/raw'  in f else None
        img_meta   = (pd.read_csv(io.StringIO(f['images/meta'][()].decode()))
                      if 'images/meta' in f else None)
        pad_size    = float(f.attrs.get('pad_size', 64))
        image_scale = float(f.attrs.get('image_scale', 1.0))
        result_dir  = Path(str(f.attrs.get('result_dir', '')))
    return df, images_raw, img_meta, pad_size, image_scale, result_dir


# ── App ───────────────────────────────────────────────────────────────────────

def build_labeler(h5_path: str, location: str = '') -> pn.viewable.Viewable:
    df, images_raw, img_meta, pad_size, image_scale, result_dir = load_h5(h5_path)

    # Old-format image fallback
    recon_images_dir = result_dir / 'recon' / 'images'
    old_img_files: list = []
    if images_raw is None and img_meta is None and result_dir != Path(''):
        old_img_files = sorted(recon_images_dir.glob('raw_*.tif'))

    pg_col   = 'patch_group' if 'patch_group' in df.columns else 'group'
    cond_col = 'condition_name' if 'condition_name' in df.columns else 'condition'

    grp_to_cond: dict = {}
    for _, row in df[[pg_col, cond_col]].dropna().drop_duplicates().iterrows():
        grp_to_cond[str(row[pg_col])] = str(row[cond_col])

    if images_raw is not None and img_meta is not None:
        unique_groups = sorted(img_meta['group'].astype(str).unique())
        def _get_canvas(group_key: str) -> np.ndarray:
            matches = img_meta[img_meta['group'].astype(str) == group_key]
            if matches.empty:
                return np.zeros((512, 512), dtype=np.float32)
            arr = images_raw[int(matches.iloc[0]['frame'])].astype(np.float32)
            mx = arr.max()
            return arr / mx if mx > 0 else arr
    else:
        unique_groups = sorted(p.stem[4:] for p in old_img_files)
        def _get_canvas(group_key: str) -> np.ndarray:
            p = recon_images_dir / f'raw_{group_key}.tif'
            arr = tifffile.imread(str(p)).astype(np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            mx = arr.max()
            return arr / mx if mx > 0 else arr

    img_options = {f"{grp_to_cond.get(g, '?')} | {g}": g
                   for g in unique_groups}

    # ── Label storage ─────────────────────────────────────────────────────────
    labels: dict[str, str] = {}   # filename → label
    _state: dict = {}

    # ── Bokeh figure ──────────────────────────────────────────────────────────
    init_group = unique_groups[0]
    init_arr   = _get_canvas(init_group)
    H, W = init_arr.shape[:2]

    img_src = ColumnDataSource(dict(
        image=[np.ascontiguousarray(np.flipud(init_arr))],
        x=[0], y=[0], dw=[W], dh=[H],
    ))
    rects_src = ColumnDataSource(dict(
        x=[], y=[], width=[], height=[], fill_color=[], fill_alpha=[],
        line_color=[], df_idx=[],
    ))
    sel_src = ColumnDataSource(dict(x=[], y=[], width=[], height=[]))

    canvas_fig = figure(
        width=720, height=720,
        x_range=Range1d(0, W), y_range=Range1d(0, H),
        title='Select a label below, then click a patch to assign it',
        tools='pan,wheel_zoom,reset,tap',
        toolbar_location='above',
    )
    gray_mapper = LinearColorMapper(palette=GRAY256, low=0.0, high=1.0)
    canvas_fig.image(
        image='image', source=img_src,
        x=0, y=0, dw=W, dh=H,
        color_mapper=gray_mapper,
    )
    canvas_fig.rect(
        'x', 'y', 'width', 'height', source=rects_src,
        fill_color='fill_color', fill_alpha='fill_alpha',
        line_color='line_color', line_width=1.5, line_alpha=1.0,
    )
    canvas_fig.rect(
        'x', 'y', 'width', 'height', source=sel_src,
        fill_alpha=0, line_color='white', line_width=2.8,
    )

    # ── Rect builder ─────────────────────────────────────────────────────────
    def _rects_for_group(group_key: str, img_H: int) -> dict:
        mask = df[pg_col].astype(str) == group_key
        sub  = df[mask]
        xs, ys, ws, hs = [], [], [], []
        fills, alphas, lines, idxs = [], [], [], []
        for i, row in sub.iterrows():
            cx = row.get('canvas_cx', np.nan)
            cy = row.get('canvas_cy', np.nan)
            ps = int(row.get('ps', 32))
            if pd.isna(cx) or pd.isna(cy):
                continue
            fname = str(row.get('filename', ''))
            lbl   = labels.get(fname, '')
            color = FA_COLOR_MAP.get(lbl, UNLABELED_COLOR)
            xs.append(float(cx) * image_scale)
            ys.append((img_H - float(cy)) * image_scale)
            ws.append(float(ps) * image_scale)
            hs.append(float(ps) * image_scale)
            fills.append(color)
            alphas.append(0.8 if lbl else 0.1)
            lines.append(color)
            idxs.append(i)
        return dict(x=xs, y=ys, width=ws, height=hs,
                    fill_color=fills, fill_alpha=alphas,
                    line_color=lines, df_idx=idxs)

    _state.update(group=init_group, H=H, W=W)
    rects_src.data = _rects_for_group(init_group, H)

    def _load_group(group_key: str) -> None:
        arr    = _get_canvas(group_key)
        Hn, Wn = arr.shape[:2]
        img_src.data = dict(
            image=[np.ascontiguousarray(np.flipud(arr))],
            x=[0], y=[0], dw=[Wn], dh=[Hn],
        )
        canvas_fig.x_range.start, canvas_fig.x_range.end = 0, Wn
        canvas_fig.y_range.start, canvas_fig.y_range.end = 0, Hn
        rects_src.data = _rects_for_group(group_key, Hn)
        sel_src.data   = dict(x=[], y=[], width=[], height=[])
        _state.update(group=group_key, H=Hn, W=Wn)

    # ── Widgets ───────────────────────────────────────────────────────────────
    img_selector = pn.widgets.Select(
        name='Image', options=img_options, value=init_group, width=440,
    )
    img_selector.param.watch(lambda e: _load_group(e.new), 'value')

    label_group = pn.widgets.RadioButtonGroup(
        name='Active label',
        options=LABEL_OPTIONS,
        value=LABEL_OPTIONS[0],
        button_type='default',
        width=560,
    )

    name_input = pn.widgets.TextInput(
        placeholder='Your name (used in save filename)…',
        width=240,
    )

    status_md = pn.pane.HTML(
        '<i style="color:#888;">Enter your name, select a label, then click a patch.</i>',
        width=560,
    )
    count_md = pn.pane.Markdown('**Labeled:** 0', width=120)

    def _update_count() -> None:
        count_md.object = f'**Labeled:** {len(labels)}'

    # ── Canvas tap handler ────────────────────────────────────────────────────
    def _on_tap(event: Tap) -> None:
        H_cur = _state['H']
        tap_cx = event.x / image_scale
        tap_cy = (H_cur - event.y) / image_scale

        xs_bk  = np.array(rects_src.data['x'],      dtype=float)
        ys_bk  = np.array(rects_src.data['y'],      dtype=float)
        df_idx = np.array(rects_src.data['df_idx'], dtype=int)
        if len(xs_bk) == 0:
            return

        cx_arr = xs_bk / image_scale
        cy_arr = (H_cur - ys_bk) / image_scale
        dists  = np.sqrt((cx_arr - tap_cx)**2 + (cy_arr - tap_cy)**2)
        near_i  = int(np.argmin(dists))
        near_df = int(df_idx[near_i])

        # Save position before refresh (indices shift after update)
        sel_x = float(xs_bk[near_i])
        sel_y = float(ys_bk[near_i])
        sel_w = float(rects_src.data['width'][near_i])
        sel_h = float(rects_src.data['height'][near_i])

        row    = df.iloc[near_df]
        fname  = str(row.get('filename', ''))
        active = label_group.value
        labels[fname] = active

        # Refresh patch colours
        rects_src.data = _rects_for_group(_state['group'], _state['H'])
        sel_src.data   = dict(x=[sel_x], y=[sel_y], width=[sel_w], height=[sel_h])

        color = FA_COLOR_MAP.get(active, '#ffffff')
        status_md.object = (
            f'<span style="font-size:13px;">'
            f'Labeled <b>{Path(fname).stem}</b> → '
            f'<span style="color:{color};font-weight:bold;">{active}</span>'
            f'</span>'
        )
        _update_count()

    canvas_fig.on_event(Tap, _on_tap)

    # ── Finish & Save ─────────────────────────────────────────────────────────
    finish_btn = pn.widgets.Button(
        name='Finish & Save', button_type='success', width=160,
    )

    def _on_finish(event) -> None:
        if not labels:
            status_md.object = '<i style="color:#e55;">No labels to save.</i>'
            return
        annotator = name_input.value.strip().replace(' ', '_') or 'unknown'
        stamp = datetime.now().strftime('%Y%m%d_%H%M')
        out = Path(h5_path).parent / f'labels_{annotator}_{stamp}.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [{'filename': fn, 'label': lbl, 'annotator': annotator}
                for fn, lbl in labels.items()]
        pd.DataFrame(rows).to_csv(out, index=False)
        status_md.object = (
            f'<span style="color:#3c3;font-size:13px;font-weight:bold;">'
            f'✓ Saved {len(labels)} labels → {out.name}</span>'
        )

    finish_btn.on_click(_on_finish)

    # ── Layout ────────────────────────────────────────────────────────────────
    toolbar = pn.Row(
        pn.pane.HTML('<b style="line-height:2.2;">Active label:</b>', width=100),
        label_group,
        pn.Spacer(width=20),
        count_md,
        pn.Spacer(width=20),
        finish_btn,
    )

    return pn.Column(
        pn.pane.HTML(
            f'<h2>Patch Labeling Tool &nbsp;·&nbsp; {Path(h5_path).name}'
            + (f' &nbsp;<span style="font-size:14px;font-weight:normal;color:#888;">in {location}</span>' if location else '')
            + '</h2>',
            sizing_mode='stretch_width',
        ),
        pn.Row(
            pn.pane.HTML('<b style="line-height:2.2;">Annotator:</b>', width=80),
            name_input,
            pn.Spacer(width=20),
            img_selector,
        ),
        toolbar,
        status_md,
        pn.pane.Bokeh(canvas_fig),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('h5', nargs='+', help='One or more interactive.h5 files')
    p.add_argument('--port', type=int, default=5007)
    p.add_argument('--serve', action='store_true',
                   help='Bind to 0.0.0.0 for network access (lab server mode). '
                        'Lab members open http://<server-ip>:<port>/<name> in their browser.')
    p.add_argument('--nas-mount', default=None,
                   help='NAS mount prefix to strip from paths, e.g. /mnt/p/')
    p.add_argument('--nas-name', default=None,
                   help='Human-readable NAS label, e.g. "GardelNas Expansion"')
    return p.parse_args()


def _get_h5_path() -> str:
    sess = pn.state.session_args
    if 'args' in sess and sess['args']:
        arg = sess['args'][0]
        return arg.decode() if isinstance(arg, bytes) else str(arg)
    if len(sys.argv) > 1:
        return sys.argv[1]
    print('Usage: python scripts/label_patches.py path/to/interactive.h5',
          file=sys.stderr)
    sys.exit(1)


if pn.state.served:
    _h5 = _get_h5_path()
    build_labeler(_h5).servable()

if __name__ == '__main__':
    import socket
    args = _parse_args()
    h5_paths = [str(Path(p).resolve()) for p in args.h5]

    def _location(h: str) -> str:
        """Build the 'in <location>' subtitle for a given H5 path."""
        p = Path(h).parent   # folder containing interactive.h5
        if args.nas_mount and args.nas_name:
            mount = args.nas_mount.rstrip('/')
            rel   = str(p).removeprefix(mount).lstrip('/')
            return f'{args.nas_name}: {rel}'
        return ''

    # Build route dict: one H5 → serve at '/', multiple → serve at '/<parent_folder>'
    if len(h5_paths) == 1:
        routes = {'/': lambda h=h5_paths[0], loc=_location(h5_paths[0]): build_labeler(h, loc)}
    else:
        routes = {f'/{Path(h).parent.name}': (lambda h=h, loc=_location(h): build_labeler(h, loc))
                  for h in h5_paths}

    if args.serve:
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(('8.8.8.8', 80))
            host_ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            host_ip = '0.0.0.0'
        print(f'[label] Serving in lab mode on port {args.port}')
        for route, h in zip(routes.keys(), h5_paths):
            print(f'[label]   http://{host_ip}:{args.port}{route}  →  {h}')
        pn.serve(routes,
                 address='0.0.0.0',
                 port=args.port,
                 allow_websocket_origin=['*'],
                 show=False,
                 autoreload=False)
    else:
        for route, h in zip(routes.keys(), h5_paths):
            print(f'[label]   http://localhost:{args.port}{route}  →  {h}')
        pn.serve(routes,
                 show=True, port=args.port, autoreload=False)
