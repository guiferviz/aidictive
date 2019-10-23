
from .base_logger import BaseLogger
from .interval_logger import IntervalLogger


_LOGGERS = {
    "interval": IntervalLogger
}


def add(name, logger):
    _LOGGERS[name] = logger

def get(name_or_object):
    if type(name_or_object) == str:
        return _LOGGERS[name_or_object]
    return name_or_object

def create(name_or_object, total_samples, **kwargs):
    # TODO: create a generic method that creates objects.
    """Create a logger using the given parameters. """

    opt = get(name_or_object)
    # If opt is a class create and return object.
    if type(opt) == type:
        return opt(total_samples, **kwargs)
    # If the object is already instanciated check that we do not have
    # unuseful parameters.
    assert not bool(kwargs)
    # Return object as is.
    return opt

