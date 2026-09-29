"""Resolve paths embedded in released validation metadata against a mounted dev_set."""

from pathlib import Path


DEV_SET_MARKER = "data/dev_set/"


def dev_set_root_from_metadata_list(metadata_list: str | Path) -> Path:
    """The validation index lives at <dev_set>/metadata/valid.json."""
    return Path(metadata_list).resolve().parent.parent


def rebase_dev_set_paths(value, data_root: str | Path):
    """Return a copy with released dev_set paths pointing at the mounted data root."""
    if isinstance(value, dict):
        return {key: rebase_dev_set_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [rebase_dev_set_paths(item, data_root) for item in value]
    if isinstance(value, str):
        if value.startswith(DEV_SET_MARKER) or (
            value.startswith("/") and DEV_SET_MARKER in value
        ):
            return str(Path(data_root) / value.split(DEV_SET_MARKER, 1)[1])
    return value
