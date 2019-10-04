
import torch

from .radam import RAdam


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "radam": RAdam,
}


def add(name, obj):
    """Add an optimizer to the library so you can use it as str. """
    if not isinstance(name_or_object, torch.optim.optimizer):
        raise Exception(f"Expecting an optimizer, get {type(obj)}")
    _OPTIMIZERS[name] = obj

def get(name_or_object):
    """Get an optimizer from name or return an object. """
    if type(name_or_object) == str:
        return _OPTIMIZERS[name]
    if isinstance(name_or_object, torch.optim.optimizer):
        return name_or_object
    raise Exception(f"Expecting an optimizer, get {type(name_or_object)}")

def create(name_or_object, **kwargs):
    """Create an optimizer using the given parameters. """
    opt = get(name_or_object)
    return opt(**kwargs)

