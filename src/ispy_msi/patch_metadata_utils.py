#!/usr/bin/env python
"""Patch-level metadata utility functions."""
from pathlib import Path


from pandas import DataFrame


def get_patch_metadata(patch_dir: Path) -> DataFrame:
    """Return the patch-level metadata in a ``DataFrame``.

    Parameters
    ----------
    patch_dir : Path
        Directory containing the patches.

    Returns
    -------
    metadata : DataFrame
        ``DataFrame`` holding the patch-level metadata.

    Raises
    ------
    RuntimeError
        If the file names of the patches and the masks don't all match.

    """
    metadata = DataFrame()

    metadata["mask_path"] = sorted(
        list((patch_dir / "patch-masks").glob("*.png")),
        key=lambda x: x.name,
    )
    metadata["patch_path"] = sorted(
        list((patch_dir / "patches").glob("*.png")),
        key=lambda x: x.name,
    )

    parent_name = metadata.patch_path.apply(lambda x: x.name.split("---")[0])
    parent_name = parent_name.apply(lambda x: f"{x}.png")

    print(parent_name)

    metadata["parent_img"] = parent_name.apply(lambda x: patch_dir / f"images/{x}")
    metadata["parent_mask"] = parent_name.apply(lambda x: patch_dir / f"masks/{x}")

    all_match = metadata.apply(
        lambda x: x.patch_path.stem == x.mask_path.stem,
        axis=1,
    ).all()

    if not all_match:
        raise RuntimeError("Mask and patch names do not match.")

    return metadata.reset_index(drop=True)
