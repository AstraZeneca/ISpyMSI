#!/usr/bin/env python
"""MSI-tissue-segmentation inference script."""
# from typing import Dict, Optional, Tuple
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from argparse import BooleanOptionalAction

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib import rcParams

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore

from skimage.io import imread
from skimage.measure import find_contours  # pylint: disable=no-name-in-module

from sklearn.metrics import f1_score  # type: ignore

import numpy as np
from numpy import ndarray


from torch import load, no_grad, Tensor, from_numpy  # pylint: disable=no-name-in-module
from torch.cuda import is_available
from torch.nn import Module

from ispy_msi.models import SegModel

# from ispy_msi.transforms import as_tensor
from ispy_msi.transforms import compose_input_tfms
from ispy_msi import plotting


plt.switch_backend("agg")
plt.style.use(Path(plotting.__file__).with_name("matplotlibrc"))

rcParams["text.usetex"] = bool(shutil.which("latex"))

DEVICE = "cuda" if is_available() else "cpu"


def _parse_command_line() -> Namespace:
    """Parse the commnad-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="MSI-tissue-segmenation inference.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "msi_path",
        help="Path to the MSI representation to predict on.",
        type=Path,
    )

    parser.add_argument(
        "checkpoint",
        help="Path to the checkpoint to infer with.",
        type=Path,
    )

    parser.add_argument(
        "mpp",
        help="Microns per pixel of the scan.",
        type=float,
    )

    parser.add_argument(
        "--out-dir",
        help="Directory to save the predicted masks in.",
        type=Path,
        default="pred-plots",
    )

    parser.add_argument(
        "--zero-background",
        help="Should we zero the background pixels in the predictions?",
        type=bool,
        default=True,
        action=BooleanOptionalAction,
    )

    return parser.parse_args()


@no_grad()
def _make_prediction(model: Module, msi_rep: Tensor) -> ndarray:
    """Use ``model`` to make a prediction on ``msi_rep``.

    Parameters
    ----------
    model : Module
        The trained segmentation model.
    msi_rep : ndarray
        The MSI representation of the MSI we want to predict on.

    Returns
    -------
    ndarray
        The predicted binary mask.

    """
    msi_rep = msi_rep.unsqueeze(0).to(DEVICE)

    return model(msi_rep).argmax(dim=1).squeeze().numpy().astype(bool)


def produce_segmentation_visual(args: Namespace):  # pylint: disable=too-many-locals
    """Compare the ground truths with the model's predictions.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    model = SegModel()
    model.load_state_dict(load(args.checkpoint, map_location="cpu"))
    model = model.to(DEVICE)

    tfms = compose_input_tfms(training=False)
    msi_rep = tfms(args.msi_path)

    mask_path = args.msi_path.parent.with_name("masks") / args.msi_path.name
    mask = imread(mask_path).astype(bool)

    pred = _make_prediction(model, msi_rep)

    dice = f1_score(pred.flatten(), mask.flatten())

    colours = iter(["g", "cyan"])
    styles = iter(["-", "--"])
    labels = ["Label", f"Prediction (Dice {np.round(dice, 2):.2f})"]

    width = 8.27
    height = (pred.shape[0] / pred.shape[1]) * (width / 2.0)

    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    for mask_img, axis, label in zip([mask, pred], axes.ravel(), labels):  # type: ignore

        plt_img = msi_rep[1, :, :]

        if "prediction" in label.lower() and args.zero_background:
            plt_img *= from_numpy(mask_img)

        axis.imshow(plt_img, cmap="inferno")

        colour = next(colours)
        style = next(styles)

        contours = find_contours(mask_img)
        for contour in contours[:-1]:
            axis.plot(contour[:, 1], contour[:, 0], style, color=colour, lw=1.5)

        axis.plot(
            contours[-1][:, 1],
            contours[-1][:, 0],
            style,
            color=colour,
            label=label,
            lw=1.5,
        )

        scalebar = AnchoredSizeBar(
            axis.transData,
            5000 / args.mpp,
            r"$5\ \mathrm{mm}$",
            "lower right",
            color="white",
            frameon=False,
            fontproperties={"size": 15},
            size_vertical=2,
        )
        axis.add_artist(scalebar)

        axis.set_xticks([])
        axis.set_yticks([])

        axis.legend(frameon=True, fontsize=13, loc="upper left")

    args.out_dir.mkdir(exist_ok=True, parents=True)

    fig.tight_layout(pad=0.05)
    fig.savefig(args.out_dir / args.msi_path.with_suffix(".pdf").name, dpi=250)


if __name__ == "__main__":
    produce_segmentation_visual(_parse_command_line())
