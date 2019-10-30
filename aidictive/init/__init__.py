
import torch

from .truncated_normal import truncated_normal_


_INIT = {
    "truncated_normal": truncated_normal_,
    "uniform_": torch.nn.init.uniform_,
}


def add(name, fun):
    _INIT[name] = fun

def get(name_or_fun):
    if type(name_or_fun) == str:
        return _INIT[name_or_fun]
    return name_or_fun

def initialize(tensor, name_or_fun, **kwargs):
    if type(name_or_fun) == dict:
        # We are going to overwrite kwargs, ensure you are not using them.
        assert not bool(kwargs)
        kwargs = name_or_fun.get("params", {})
        name_or_fun = name_or_fun["name"]
    init_fun = get(name_or_fun)
    init_fun(tensor, **kwargs)

