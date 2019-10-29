
import torch


_SCHEDULERS = {
    # I've never remember the exact name, so we add 3 times with different
    # names the scheduler ReduceLROnPlateau.
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "reduceonplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def add(name, obj):
    """Add an scheduler to the library so you can use it as str. """

    _SCHEDULERS[name] = obj


def get(name_or_object):
    """Get an scheduler from name or return an object. """

    if type(name_or_object) == str:
        return _SCHEDULERS[name_or_object]
    return name_or_object


def create(name_or_object, optimizer, **kwargs):
    """Create an scheduler using the given parameters. """

    opt = get(name_or_object)
    return opt(optimizer, **kwargs)

