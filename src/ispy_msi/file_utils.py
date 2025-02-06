"""File-processing utilities."""

from pathlib import Path

from pandas import DataFrame  # type: ignore


def list_msi_data(directory: Path) -> DataFrame:
    """List the MSI data in ``dir``.

    Parameter
    ---------
    directory : Path
        Directory holding the data.

    Returns
    -------
    files : DataFrame
        A ``DataFrame`` holding the MSI data files.

    Raises
    ------
    TypeError
        If ``dir`` is not a Path.
    TypeError
        If ``dir` is not an existing directory.
    FileNotFounderror
        If we don't find a ``".ibd"`` for each ``".imzml"`` file.

    """
    if not isinstance(directory, Path):
        msg = f"Argument '{directory}' should be a directory, got"
        msg += f"'{type(directory)}'."
        raise TypeError(msg)

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    files = DataFrame(columns=["imzml"], data=list(directory.glob("*.imzML")))
    files["ibd"] = files.imzml.apply(lambda x: x.with_suffix(".ibd"))

    if not files.ibd.apply(lambda x: x.is_file()).all():
        raise FileNotFoundError(f"Missing '.ibd' files in '{files}'.")

    return files
