#!/usr/bin/env python
"""Train a model to detect foreground in total-ion-current images."""
from typing import Dict, Any, List, Optional
from pathlib import Path
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from time import perf_counter

from pandas import DataFrame
from numpy import nanmean, nan, linspace

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # type: ignore

from torch import (  # pylint: disable=no-name-in-module
    set_grad_enabled,
    save,
    no_grad,
    Tensor,
)
from torch.utils.data import DataLoader
from torch.cuda import is_available
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from torch.optim.lr_scheduler import PolynomialLR

from torch_tools import DataSet


from ispy_msi.patch_metadata_utils import get_patch_metadata
from ispy_msi.transforms import (
    compose_input_tfms,
    compose_target_tfms,
    compose_both_tfms,
)

from ispy_msi.models import SegModel

from ispy_msi.plotting import (
    plot_metrics_cv,
    plot_metrics,
    visualise_batches,
)

# pylint: disable=too-many-locals


DEVICE = "cuda" if is_available() else "cpu"


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments in a ``Namespace``.

    """
    parser = ArgumentParser(description="Train semantic segmentation model.")

    parser.add_argument(
        "patch_dir", help="Directory containing the patches and masks.", type=Path
    )

    parser.add_argument(
        "--output-dir",
        help="Directory to save the segmentation data in.",
        type=Path,
        default="segmentation-data",
    )

    parser.add_argument(
        "--workers",
        help="Number of workers in the data loader",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--epochs",
        help="Number of epochs to train for.",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--cv-folds",
        help="Number of cross-validation folds.",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--lr",
        help="Learning rate.",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--wd",
        help="Weight decay.",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--bs",
        help="Batch size.",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--instance-norm",
        help="Should the model use instance normalisation.",
        type=bool,
        default=True,
        action=BooleanOptionalAction,
    )

    parser.add_argument(
        "--train-all",
        help="If true, train on all of the data without validation.",
        type=bool,
        default=False,
        action=BooleanOptionalAction,
    )

    parser.add_argument(
        "--save-batches",
        help="The number of batch visualisations to save to file.",
        type=int,
        default=25,
    )

    return parser.parse_args()


def _create_dataloader(
    image_paths: List[Path],
    mask_paths: List[Path],
    batch_size: int,
    workers: int,
    training: bool = False,
):
    """Create a ``DataLoader``.

    Parameters
    ----------
    image_paths : List[Path]
        List of paths to the image files.
    mask_paths : List[Path]
        List of paths to the mask files.
    batch_size : int
        Mini-batch size.
    workers : int
        Number of workers the dataloader should use.
    training : bool, optional
        Training or validating?

    Returns
    -------
    DataLoader
        Image-mask-yielding data loader.

    """
    data_set = DataSet(
        inputs=image_paths,
        targets=mask_paths,
        input_tfms=compose_input_tfms(training=training),
        target_tfms=compose_target_tfms(),
        both_tfms=compose_both_tfms(training=training),
        mixup=training,
    )

    return DataLoader(
        data_set,
        shuffle=training,
        num_workers=workers,
        batch_size=batch_size,
        drop_last=False,
    )


def _create_pytorch_objects(
    learn_rate: float,
    weight_decay: float,
    instance_norm: bool,
    epochs: int,
) -> Dict[str, Any]:
    """Create the Pytorch-specific objects neccessary for training.

    Parameters
    ----------
    learn_rate : float
        The learning rate to use during training.
    weight_decay : float
        The weight decay to use during training.
    instance_norm : bool
        Whether or not to use instance normalisation over batch norm.
    epochs : int
        The number of training epochs.

    Returns
    -------
    Dict[str, Any]
        Pytorch machinery.

    """
    model = SegModel(instance_norm=instance_norm).to(DEVICE)
    print(f"Model device: '{DEVICE}'.")

    optimiser = Adam(
        model.parameters(),
        lr=learn_rate,
        weight_decay=weight_decay,
    )

    scheduler = PolynomialLR(
        optimizer=optimiser,
        total_iters=epochs,
        power=1.0,
    )

    return {"model": model, "optimiser": optimiser, "scheduler": scheduler}


@no_grad()
def compute_metrics(preds: Tensor, targets: Tensor) -> Dict[str, float]:
    """Compute the agreement metrics between ``preds`` and ``targets``.

    Parameters
    ----------
    preds : Tensor
        The model's predictions.
    targets : Tensor
        The ground truths.

    Returns
    -------
    Dict[str, float]
        A dictionary of the agreement metrics.

    """
    preds = preds.cpu().argmax(dim=1).flatten()
    targets = targets.cpu().argmax(dim=1).flatten()
    return {
        "accuracy": accuracy_score(targets, preds),
        "precision": precision_score(targets, preds, zero_division=nan),
        "recall": recall_score(targets, preds, zero_division=nan),
        "dice": f1_score(targets, preds, zero_division=nan),
    }


def one_epoch(
    model: SegModel,
    data_loader: DataLoader,
    optimiser: Optional[Adam] = None,
):
    """Train the model for a single epoch.

    Parameters
    ----------
    model : UNet
        Model to train.
    data_loader : DataLoader
        Training ``DataLoader``.
    optimiser : Adam, optional
        Adam optimiser.

    Returns
    -------
    Dict[str, float]
        Performance metrics.

    """
    metrics: Dict[str, List[float]] = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "dice": [],
    }

    _ = model.train() if optimiser is not None else model.eval()

    for batch, targets in data_loader:
        batch, targets = batch.to(DEVICE), targets.to(DEVICE)

        if optimiser is not None:
            optimiser.zero_grad()

        with set_grad_enabled(optimiser is not None):
            preds = model(batch)

        loss = binary_cross_entropy(preds, targets)

        if optimiser is not None:
            loss.backward()
            optimiser.step()

        metrics["loss"].append(loss.item())
        for key, val in compute_metrics(preds, targets).items():
            metrics[key].append(val)

    return {key: nanmean(val) for key, val in metrics.items()}


def _save_metrics(csv_file: Path, metrics: DataFrame):
    """Save the metrics to file.

    Parameters
    ----------
    csv_file : Path
        Path to the csv file in question.
    metrics : DataFrame
        The training (and validation) metrics.

    """
    if not csv_file.exists():
        metrics.to_csv(csv_file, index=False)
    else:
        metrics.to_csv(csv_file, index=False, mode="a", header=False)


def train_one_model(  # pylint: disable=too-many-arguments
    train_df: DataFrame,
    valid_df: DataFrame,
    args: Namespace,
    save_dir: Path,
    fold: str,
):
    """Train a single model.

    Parameters
    ----------
    train_df : DataFrame
        Metadata for the training patches.
    valid_df : DataFrame
        Metadata for the validation patches.
    args : Namespace
        Command-line arguments.
    save_dir : Path
        Directory to save the output data in.
    fold: int
        The name of the cross-valid fold.

    """
    print(f"There are {len(train_df)} images in fold {fold}.")
    print(f"There are {len(valid_df)} images in fold {fold}.")

    train_loader = _create_dataloader(
        train_df.patch_path.to_list(),
        train_df.mask_path.to_list(),
        args.bs,
        args.workers,
        training=True,
    )
    valid_loader = _create_dataloader(
        valid_df.parent_img.to_list(),
        valid_df.parent_mask.to_list(),
        args.bs,
        args.workers,
        training=False,
    )

    torch_objects = _create_pytorch_objects(
        args.lr,
        args.wd,
        args.instance_norm,
        args.epochs,
    )

    print(torch_objects["model"])

    train_metrics, valid_metrics = [], []

    for epoch in range(args.epochs):
        train_metrics.append(
            one_epoch(
                torch_objects["model"],
                train_loader,
                optimiser=torch_objects["optimiser"],
            )
        )

        valid_metrics.append(
            one_epoch(
                torch_objects["model"],
                valid_loader,
            )
        )

        torch_objects["scheduler"].step()
        print(torch_objects["scheduler"].get_last_lr())

        save(
            torch_objects["model"].state_dict(),
            save_dir / f"checkpoint-{epoch}.pth",
        )

    metrics = DataFrame(train_metrics).join(
        DataFrame(valid_metrics),
        lsuffix="_train",
        rsuffix="_valid",
    )
    metrics["fold"] = fold
    metrics["epoch"] = linspace(1, len(metrics), len(metrics))

    csv_file = save_dir.parent / "metrics.csv"
    _save_metrics(csv_file, metrics)
    plot_metrics_cv(save_dir.parent / "metrics.csv")

    visualise_batches(
        train_loader,
        save_dir / "batch-imgs/train",
        args.save_batches,
    )
    visualise_batches(
        valid_loader,
        save_dir / "batch-imgs/valid",
        args.save_batches,
    )


def train_without_validation(args: Namespace, train_df: DataFrame):
    """Train on all of the data without validation.

    Parameters
    ----------
    train_df : Dataframe
        Training data.
    args : Namespace
        Command-line arguments.

    """
    save_dir = args.output_dir / "train-all"
    save_dir.mkdir(exist_ok=True, parents=True)

    train_df.to_csv(save_dir / "metadata.csv", index=False)

    train_loader = _create_dataloader(
        train_df.patch_path.to_list(),
        train_df.mask_path.to_list(),
        args.bs,
        args.workers,
        training=True,
    )

    torch_objects = _create_pytorch_objects(
        args.lr,
        args.wd,
        args.instance_norm,
        args.epochs,
    )

    print(torch_objects["model"])

    train_metrics = []

    for epoch in range(args.epochs):
        train_metrics.append(
            one_epoch(
                torch_objects["model"],
                train_loader,
                torch_objects["optimiser"],
            )
        )

        torch_objects["scheduler"].step()
        print(torch_objects["scheduler"].get_last_lr())

        save(
            torch_objects["model"].state_dict(),
            save_dir / f"checkpoint-{epoch}.pth",
        )

    csv_file = save_dir / "metrics.csv"
    _save_metrics(csv_file, DataFrame(train_metrics))

    plot_metrics(csv_file)

    visualise_batches(
        train_loader,
        save_dir / "batch-/train/",
        args.save_batches,
    )


def train_cv_folds(args: Namespace, metadata: DataFrame):
    """Train all cross-validation folds.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.
    metadata : DataFrame
        The patch-level metadata.

    """
    args.output_dir /= "cv"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    print(metadata)

    metadata.to_csv(args.output_dir / "metadata.csv", index=False)

    fold_names = (
        metadata.fold.drop_duplicates().sort_values().sample(frac=1.0, random_state=666)
    )

    for fold in fold_names:

        save_dir = args.output_dir / f"{fold}/"
        save_dir.mkdir(exist_ok=True, parents=True)

        valid_df = metadata.loc[metadata.fold == fold]
        valid_df = valid_df.drop_duplicates(subset="parent_img")

        train_df = metadata.loc[metadata.fold != fold]

        train_df.to_csv(save_dir / "train_split.csv", index=False)
        valid_df.to_csv(save_dir / "valid_split.csv", index=False)

        start = perf_counter()

        train_one_model(train_df, valid_df, args, save_dir, fold)  # type: ignore

        stop = perf_counter()

        print(f"Fold '{fold}' time: {stop - start:.6f} seconds.")


def _add_cv_folds(patch_metadata: DataFrame):
    """Add the cross validation fold to the patch metadata.

    Parameters
    ----------
    patch_metadata : DataFrame
        Patch-level metadata.

    Returns
    -------
    DataFrame
        ``patch_metadata`` with the folds added.

    """
    patch_metadata["fold"] = patch_metadata.parent_img.apply(lambda x: x.stem).to_list()


def train_all_configs(args: Namespace):
    """Train al of the model configurations.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    args.output_dir.mkdir(exist_ok=True, parents=True)

    patch_metadata = get_patch_metadata(args.patch_dir)

    print(patch_metadata)
    print(f"There are {patch_metadata.parent_img.nunique()} MSIs.")

    if not args.train_all:
        _add_cv_folds(patch_metadata)
        print(f"There are {patch_metadata.fold.nunique()} CV folds.")
        train_cv_folds(args, patch_metadata)
    else:
        train_without_validation(args, patch_metadata)


if __name__ == "__main__":
    train_all_configs(_parse_command_line())
