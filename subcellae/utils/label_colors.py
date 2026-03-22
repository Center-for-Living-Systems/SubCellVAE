"""
label_colors.py
===============
Project-wide label orders and colour mappings for FA-type and position labels.

Import anywhere with::

    from subcellae.utils.label_colors import (
        classification_label_order,
        classification_label_to_color,
        position_label_order,
        position_label_to_color,
    )
"""

TAB10_COLORS = [
    "#1f77b4",  # 0 - blue
    "#ff7f0e",  # 1 - orange
    "#2ca02c",  # 2 - green
    "#d62728",  # 3 - red
    "#9467bd",  # 4 - purple
    "#8c564b",  # 5 - brown
    "#e377c2",  # 6 - pink
    "#7f7f7f",  # 7 - gray
    "#bcbd22",  # 8 - olive
    "#17becf",  # 9 - cyan
]

# ---------------------------------------------------------------------------
# FA-type classification labels
# ---------------------------------------------------------------------------

classification_label_order = [
    "Nascent Adhesion",
    "focal complex",
    "focal adhesion",
    "fibrillar adhesion",
    "No adhesion",
    "Uncertain",
]

classification_label_to_color = {
    cls: TAB10_COLORS[i]
    for i, cls in enumerate(classification_label_order)
}

fa_label_to_id = {
    cls: i
    for i, cls in enumerate(classification_label_order)
}

# ---------------------------------------------------------------------------
# Position labels
# ---------------------------------------------------------------------------

position_label_order = [
    "Cell Protruding Edge",
    "Cell Periphery/other",
    "Lamella",
    "Cell Body",
    "No Category/uncertain",
]

position_label_to_color = {
    cls: TAB10_COLORS[i]
    for i, cls in enumerate(position_label_order)
}

position_label_to_id = {
    cls: i
    for i, cls in enumerate(position_label_order)
}
