"""Module to house the sementation models."""

from torch import Tensor
from torch.nn import Module, BatchNorm1d, BatchNorm2d, InstanceNorm1d, InstanceNorm2d


from torch_tools import UNet


class SegModel(Module):
    """Semantic segmentation model.

    Parameters
    ----------
    instance_norm : bool, optional
        Whether or not to use instance norm instead of batch norm.

    """

    def __init__(self, instance_norm: bool = True):
        """Build ``SegModel``."""
        super().__init__()
        self._model = UNet(
            in_chans=3,
            out_chans=2,
            block_style="conv_res",
            num_layers=6,
            pool_style="avg",
            bilinear=True,
            dropout=0.0,
        )

        if instance_norm is True:
            self.apply(_batch_norm_to_instance_norm)

    def forward(self, batch: Tensor) -> Tensor:
        """Pass ``batch`` through the model.

        Parameters
        ----------
        batch : Tensor
            A mini-batch of inputs.

        Returns
        -------
        Tensor
            The result of passing ``batch`` through the model.

        """
        return self._model(batch).softmax(dim=1)


def _batch_norm_to_instance_norm(layer: Module):
    """Turn the batch normalisation layers to instance normalisations.

    Parameters
    ----------
    layer : Module
        Layer in the network.

    """
    for name, module in layer.named_children():

        if isinstance(module, BatchNorm1d):
            setattr(layer, name, InstanceNorm1d(module.num_features))

        if isinstance(module, BatchNorm2d):
            setattr(layer, name, InstanceNorm2d(module.num_features))
