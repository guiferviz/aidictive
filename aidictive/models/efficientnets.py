
import warnings

import efficientnet_pytorch

import torch

from aidictive.models.utils import freeze, unfreeze


def efficientnet(n_outputs, b="0", pretrained=False, in_channels=3,
                 freeze=False):
    # Create the model.
    net_name = f"efficientnet-b{b}"
    if pretrained:
        model = efficientnet_pytorch.EfficientNet.from_pretrained(net_name)
    else:
        model = efficientnet_pytorch.EfficientNet.from_name(net_name)

    # Freeze the params if needed.
    if freeze:
        # It does not make sense to freeze a non-pretrained model.
        assert pretrained
        freeze(model)
        # Method for easily unfreeze model.
        model.unfreeze = lambda: unfreeze(model)

    if in_channels != 3:
        # Change the first conv layer to get the desired number of in_channels.
        if pretrained:
            warnings.warn("Replacing the first convolutional layer of the "
                          "pretrained network to fit the given input "
                          "channels.")
        # Get the number of output channels of the first convolution.
        out_channels_conv_stem = model._conv_stem.out_channels
        # Recreate the layer using the parameters in the repository.
        # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        # TODO: copy parameters from the existing conv.
        model._conv_stem = torch.nn.Conv2d(in_channels, out_channels_conv_stem,
                kernel_size=3, stride=2, bias=False)
    # Get the input number of features of the last linear layer.
    in_fc_features = model._fc.in_features
    # Replace the last fc layer to get the desired number of outputs.
    model._fc = torch.nn.Linear(in_fc_features, n_outputs)
    return model

def efficientnet_b0(n_outputs, **kwargs):
    return efficientnet(n_outputs, b="0", **kwargs)

