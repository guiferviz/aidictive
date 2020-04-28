
from aidictive.models.utils import freeze, unfreeze

from aidictive.models.ffnn import FFNN
from aidictive.models.resnets import resnet, resnet18


################################
#  Non-required dependencies.  #
################################
# Raise error only if the non-required models are used.

def _set_error(model, package):
    return lambda *arg, **kwargs: _show_error(model, package)

def _show_error(model, package):
    raise Exception(f"You need to install the package '{package}' to use "
                     "the '{model}' model")

try:
    from aidictive.models.efficientnets import efficientnet, efficientnet_b0
except:
    model, package = "EfficientNet", "efficientnet_pytorch"
    efficientnet = _set_error(model, package)
    efficientnet_b0 = _set_error(model, package)

