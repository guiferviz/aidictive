
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
    """Get an activation function from name or return object.
 
    The idea of returning and object allows us to use string or objects in
    any place where an activation function is required.
    Anyways, the cleanest way of using personalized activation functions is to
    add them to the list of available activations functions and use the given
    name.
    The only requisite for this object is that it should be callable because
    the rest of the AIdictive code is going to use it as an activation
    function.
    This function does not check if the object is callable, only returns the
    object.
    """

    if type(name_or_fun) == str:
        return _ACTIVATIONS[name_or_fun]
    return name_or_fun

