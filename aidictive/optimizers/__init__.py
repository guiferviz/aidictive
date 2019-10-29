
import torch

from .radam import RAdam


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "radam": RAdam,
}


def add(name, obj):
    """Add an optimizer to the library so you can use it as str.
    
    raise: Exception if the given object is not a `torch.optim.optimizer`.
    """

    if not isinstance(name, torch.optim.Optimizer):
        raise Exception(f"Expecting an optimizer, get {type(obj)}")
    _OPTIMIZERS[name] = obj


def get(name_or_object):
    """Get an optimizer from name or return an object. """

    if type(name_or_object) == str:
        return _OPTIMIZERS[name_or_object]
    if isinstance(name_or_object, torch.optim.Optimizer):
        return name_or_object
    raise Exception(f"Expecting an optimizer, get {type(name_or_object)}")


def create(name_or_object, parameters, **kwargs):
    """Create an optimizer using the given parameters.
    
    TODO: Do not use all the kwargs, get valid parameters from __init__.
    This may be useful for changing from one optimizer from another easily.
    In that way we can keep the optimizer specific parameters in the code
    and we do not need to change all the parameters when changing optimizers.
    Maybe a warning saying that some parameters are not going to be used is
    a good option.
    """

    opt = get(name_or_object)
    return opt(parameters, **kwargs)

