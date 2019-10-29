
import torch


_LOSSES = {
    "mae": torch.nn.functional.l1_loss,
    "mse": torch.nn.functional.mse_loss,
    "cross_entropy": torch.nn.functional.cross_entropy,
}


def add(name, fun):
    """Add a loss function to the library so you can use it as str. """
    _LOSSES[name] = fun


def get(name_or_fun):
    """Get a loss function from name or return object. """

    if type(name_or_fun) == str:
        return _LOSSES[name_or_fun]
    return name_or_fun

