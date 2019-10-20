
import torch


_ACTIVATIONS = {
    "sigmoid": torch.nn.Sigmoid(),
    "tanh": torch.nn.Tanh(),
    "relu": torch.nn.ReLU(),
    "leakyrelu": torch.nn.LeakyReLU(),
    "selu": torch.nn.SELU(),
    "linear": None,
}


def add(name, fun):
    """Add an activation function to the library so you can use it as str. """
    _ACTIVATIONS[name] = fun

def get(name_or_fun):
    """Get an activation function from name or return callable. """
    if type(name_or_fun) == str:
        return _ACTIVATIONS[name_or_fun]
    return name_or_fun

