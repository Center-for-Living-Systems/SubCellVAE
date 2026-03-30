"""
config_utils.py
===============
Shared helpers for YAML config loading.
"""

from __future__ import annotations


def resolve_root(raw: dict, root_override: str | None = None) -> dict:
    """Expand ``root_folder + "/..."`` expressions throughout a raw YAML dict.

    Parameters
    ----------
    raw:
        The dict returned by ``yaml.safe_load``.
    root_override:
        If provided (e.g. from ``--root_folder`` CLI flag), this value is used
        as the root and takes precedence over any ``root_folder`` key inside the
        YAML.  Pass ``None`` to fall back to the YAML key.

    Returns
    -------
    dict
        A new dict with all ``root_folder + "/..."`` strings replaced by the
        resolved absolute path.
    """
    root = root_override or raw.get("root_folder", "") or raw.get("paths", {}).get("root_folder", "")

    def _resolve(val):
        if (
            isinstance(val, str)
            and val.startswith('root_folder + "')
            and val.endswith('"')
        ):
            suffix = val[len('root_folder + "'):-1]
            return root + suffix
        return val

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return _resolve(obj)

    return _walk(raw)
