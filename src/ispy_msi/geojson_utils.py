"""Utility functions for working with QuPath-exported geojsons."""

from pathlib import Path

from pandas import DataFrame, read_json

from shapely import MultiPolygon, Polygon  # type: ignore


def prepare_roi(json_path: Path) -> DataFrame:
    """Extract patch metadata from the geojson files.

    Parameters
    ----------
    json_path : Path
        Path to geojson file.

    Returns
    -------
    DataFrame
        The metadata for the ROIs in the file at ``json_path``.

    Notes
    -----
    The geojson files contain the patient IDs, polygons delineating the
    objects in the masks and the sample status (tumour, benign, mxed). These
    data are all connected, so there is one patient ID and tumour status per
    polygon.

    """
    geo = DataFrame(read_json(json_path))

    extracted_data = []

    for row in geo.itertuples():

        if row.geometry["type"] == "Polygon":  # type: ignore
            poly = MultiPolygon([Polygon(row.geometry["coordinates"][0])])  # type: ignore
        elif row.geometry["type"] == "MultiPolygon":  # type: ignore
            poly = MultiPolygon([Polygon(x[0]) for x in row.geometry["coordinates"]])  # type: ignore # pylint: disable=line-too-long
        else:
            raise RuntimeError("Unexpected polygon object")

        row_dict = {}

        row_dict["roi"] = poly  # type: ignore
        row_dict["object_id"] = row.id  # type: ignore
        row_dict["wsi"] = json_path.stem

        extracted_data.append(row_dict)

    return DataFrame(extracted_data)
