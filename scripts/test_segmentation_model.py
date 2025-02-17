#!/usr/bin/env python
"""Test trained segmentation models."""

from typing import Dict, List

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import DataFrame

from torch import load, no_grad, Tensor
from torch.cuda import is_available

from numpy import ndarray, concat

from scipy.stats import mode  # type: ignore

from skimage.io import imread

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # type: ignore

from ispy_msi.models import SegModel
from ispy_msi.transforms import compose_input_tfms

DEVICE = "cuda" if is_available() else "cpu"


_metric_functions = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score,
}


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Parse the command-line arguments.

    """
    parser = ArgumentParser(
        description="Test the trained segmentation models.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "cv_checkpoints",
        help="Directory holding the cross-validation checkpoints.",
        type=Path,
    )

    parser.add_argument(
        "image_dir",
        help="Directory holding the images.",
        type=Path,
    )

    parser.add_argument(
        "--out-dir",
        help="Directory to save the predictions in.",
        type=Path,
        default="test-results",
    )

    return parser.parse_args()


def _get_metrics(pred: ndarray, target: ndarray) -> Dict[str, float]:
    """Return the segmentation performance metrics.

    Parameters
    ----------
    pred : ndarray
        The predicted mask.
    target : ndarray
        The target mask.

    Returns
    -------
    Dict[str, float]
        The performance metrics

    """
    pred = pred.flatten()
    target = target.flatten()

    return {
        "accuracy": accuracy_score(target, pred),
        "precision": precision_score(target, pred),
        "recall": recall_score(target, pred),
        "dice": f1_score(target, pred),
    }


@no_grad()
def infer_on_single_image(
    checkpoints: List[Path],
    img: Tensor,
    target: ndarray,
) -> List[Dict[str, float]]:
    """Infer on a single image which each of the checkpoints.

    Parameters
    ----------
    checkpoints : List[Path]
        Paths to the model checkpoints.
    img : Tensor
        The input image to test on.
    target : ndarray
        The binary segmentation mask.

    Returns
    -------
    metrics_list : List[Dict[str, float]]
        A list of dictionaries of the performance metrics for each checkpoint
        of the model.

    """
    model = SegModel()

    pred_list: List[ndarray] = []
    metrics_list: List[Dict[str, float]] = []

    for checkpoint in checkpoints:

        state_dict = load(checkpoint, weights_only=True, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        model.to(DEVICE)

        print(f"Loaded checkpoint '{checkpoint.parent.name}'")

        pred = model(img.to(DEVICE)).cpu().argmax(dim=1).squeeze().numpy()
        pred_list.append(pred)

        metrics = _get_metrics(pred, target)
        metrics["fold"] = checkpoint.parent.name  # type: ignore
        metrics_list.append(metrics)

    conced = concat(list(map(lambda x: x[:, :, None], pred_list)), axis=2)
    modal_pred = mode(conced, axis=2).mode.astype(bool)

    metrics_list.append(_get_metrics(modal_pred, target))
    metrics_list[-1]["fold"] = "mode"  # type: ignore

    return metrics_list


def test_models(args: Namespace):
    """Test the segmentation models.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    checkpoints = list(args.cv_checkpoints.glob("*/*.pth"))
    args.out_dir.mkdir(parents=True)

    print(f"There are '{len(checkpoints)}' cross-validation checkpoints.")

    metadata = DataFrame(
        columns=["img_path"],
        data=list(args.image_dir.glob("*")),
    )

    metadata["mask_path"] = metadata.img_path.apply(
        lambda x: str(x).replace("/images/", "/masks/")
    )

    input_tfms = compose_input_tfms(training=False)

    metrics_path = args.out_dir / "metrics.csv"

    for row in metadata.itertuples():

        img = input_tfms(row.img_path).unsqueeze(0)
        tgt = imread(row.mask_path).clip(0, 1)

        metrics_list = infer_on_single_image(checkpoints, img, tgt)

        metrics = DataFrame(metrics_list)
        metrics["msi"] = row.img_path.stem  # type: ignore

        if not metrics_path.exists():
            metrics.to_csv(metrics_path, index=False)
        else:
            metrics.to_csv(
                metrics_path,
                index=False,
                mode="a",
                header=False,
            )


if __name__ == "__main__":
    test_models(_parse_command_line())
