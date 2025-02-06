"""Plotting utilities."""

from typing import Union

from pathlib import Path
import shutil

from skimage.measure import find_contours  # pylint: disable=no-name-in-module
from skimage.util import img_as_ubyte

from pandas import read_csv  # type: ignore
from numpy import linspace, array, ndarray, sqrt

from torch.utils.data import DataLoader
from torchvision.utils import make_grid  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
from matplotlib.axes import Axes

from matplotlib import rcParams

plt.switch_backend("agg")
plt.style.use(Path(__file__).with_name("matplotlibrc"))

rcParams["text.usetex"] = bool(shutil.which("latex"))

_colors = array([(67, 162, 202), (221, 28, 119)]) / 255.0


def plot_metrics(csv_file: Path):
    """Plot the metrics when using 100% of the training data.

    Parameters
    ----------
    csv_file : Path
        Path to the file containing the metric to plot.

    """
    data = read_csv(csv_file)

    metrics = ["loss", "accuracy", "precision", "recall", "dice"]

    figure, axes = plt.subplots(1, len(metrics), figsize=(10, 2.0))

    for metric, axis in zip(metrics, axes.ravel()):  # type: ignore
        axis.plot(data[metric], "-o", color=_colors[1], lw=0.5, ms=2, mew=0.5)
        axis.set_ylabel(f"{metric.capitalize()}")

    for axis in axes.ravel():  # type: ignore
        axis.set_ylim(bottom=0.0, top=1.0)
        axis.set_yticks(linspace(0.0, 1.0, 6))
        axis.set_xlim(left=0.0)
        axis.set_xlabel("Epoch", labelpad=0.0)
        _square_aspect(axis)

    figure.tight_layout(pad=0.1)
    figure.savefig(csv_file.with_suffix(".pdf"), dpi=750)
    plt.close("all")


def plot_metrics_cv(csv_file: Path):  # pylint: disable=too-many-locals
    """Plot the training and validation metrics in ``csv_file``.

    Parameters
    ----------
    csv_file : Path
        Path to the csv file containing the metrics to plot.

    """
    data = read_csv(csv_file).sort_values(by=["fold", "epoch"])

    num_folds = data.fold.nunique()
    x_vals = sorted(data.epoch.unique())

    metrics = ["loss", "accuracy", "precision", "recall", "dice"]
    figure, axes = plt.subplots(1, len(metrics), figsize=(10, 2.0))

    for axis, metric in zip(axes, metrics):  # type: ignore
        for colour, style, split in zip(_colors, ["-o", "--s"], ["train", "valid"]):
            key = f"{metric}_{split}"

            all_folds = data[key].to_numpy().reshape(num_folds, len(x_vals))

            mean = all_folds.mean(axis=0)
            std = all_folds.std(axis=0)

            axis.plot(
                x_vals,
                mean,
                style,
                color=colour,
                label=split.capitalize(),
                ms=2,
            )
            axis.fill_between(
                x_vals,
                mean + (0.5 * std),
                mean - (0.5 * std),
                color=colour,
                alpha=0.25,
                label="_nolegend_",
                lw=0.0,
            )

        axis.set_ylabel(metric.capitalize())

    for axis in axes.ravel():  # type: ignore
        axis.legend(fontsize=7)
        axis.set_ylim(bottom=0.0, top=1.0)
        axis.set_yticks(linspace(0.0, 1.0, 11))
        axis.set_xlim(left=1, right=data.epoch.max())
        axis.set_xlabel("Epoch", labelpad=0.0)
        _square_aspect(axis)

    figure.tight_layout(pad=0.1)
    figure.savefig(csv_file.with_suffix(".pdf"), dpi=750)
    plt.close("all")


def plot_mask_and_prediction(
    image: ndarray,
    mask: ndarray,
    pred: ndarray,
    save_path: Path,
):
    """Plot the mask and prediction on ``image``.

    Parameters
    ----------
    image : ndarray
        The image (C, H, W) ``prediction`` and ``mask`` correspond to.
    mask : ndarray
        The 2D binary ground truth.
    pred : ndarray
        The 2D predicted mask.
    save_path : Path
        Target path to save the image at.

    """
    height, width = mask.shape
    scale = 0.01

    figure, axis = plt.subplots(1, 1, figsize=(scale * width, scale * height))

    axis.imshow(img_as_ubyte(image))
    axis.set_xticks([])
    axis.set_yticks([])

    for binary, color, line in zip([mask, pred], _colors, ["-", "--"]):
        for coords in find_contours(binary):
            axis.plot(coords[:, 1], coords[:, 0], line, color=color, lw=1)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    figure.tight_layout(pad=0.01)
    figure.savefig(save_path, dpi=500)
    plt.close("all")


def visualise_batches(loader: DataLoader, save_dir: Path, num_batches: int):
    """Write ``num_batches`` of image batches to file.

    Parameters
    ----------
    loader : DataLoader
        The batch-yielding data loader.
    save_dir : Path
        Directory to save the batches in.
    num_batches : int
        The number of batches to save to file.

    """
    save_dir.mkdir(exist_ok=True, parents=True)

    counter = 0

    for batch, targets in loader:
        img = make_grid(batch, nrow=int(sqrt(len(batch)))).permute(1, 2, 0)
        tgt = make_grid(targets, nrow=int(sqrt(len(batch)))).permute(1, 2, 0)

        figure, axis = plt.subplots(1, 1, figsize=(2.5, 2.5))

        axis.imshow(img_as_ubyte(img.clip(0.0, 1.0)))

        for coords in find_contours(tgt.argmax(dim=2).numpy()):
            axis.plot(coords[:, 1], coords[:, 0], lw=0.5, color=_colors[1])

        axis.set_xticks([])
        axis.set_yticks([])

        figure.tight_layout(pad=0.01)
        figure.savefig(save_dir / f"{counter}.pdf", dpi=500)

        figure_cleanup(axis)

        if not counter < num_batches:
            break

        counter += 1


def _square_aspect(axis: plt.Axes):
    """Make the aspect ratio square.

    Parameters
    ----------
    axis : plt.Axes
        The axis whose aspect ratio is to be set.

    """
    x_lims = axis.get_xlim()
    y_lims = axis.get_ylim()

    axis.set_aspect(abs(x_lims[0] - x_lims[1]) / abs(y_lims[0] - y_lims[1]))


def figure_cleanup(axes: Union[ndarray, Axes]):
    """Clean-up after matplotlib.

    Parameters
    ----------
    axes : ndarray
        Array of axes.

    """
    if isinstance(axes, Axes):
        axes.cla()
        axes.clear()
    else:
        for axis in axes.ravel():
            axis.cla()
            axis.clear()

    plt.close("all")


if __name__ == "__main__":
    plot_metrics_cv(Path("output-data", "cv", "metrics.csv"))
