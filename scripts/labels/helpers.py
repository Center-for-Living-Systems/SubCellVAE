import pandas as pd


def ctrl_Y_ID_adding(df):
    """
    Extract group (control/ycomp) from czi_filename and add unique_ID.
    Adds columns: group, group_ID (0=control, 1=ycomp), unique_ID.
    """
    df["group"] = (
        df["czi_filename"]
        .str.lower()
        .str.extract(r"(control|ycomp)", expand=False)
    )
    df["group_ID"] = df["group"].map({"control": 0, "ycomp": 1})
    df["unique_ID"] = df["group"] + '-' + df["crop_img_filename"]
    return df


LABEL_COLS = ["unique_ID", "crop_img_filename", "group", "group_ID", "Position", "classification"]