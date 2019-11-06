
import torch
import torchvision

from aidictive.models.utils import freeze, unfreeze


_RESNETS = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152,
}


def resnet(n_outputs, n="18", in_channels=3, pretrained=False, freeze=False):
    """Create a resnet model using torchvision.

    You can ajust the number of input_channels and the number of outputs.
    Please, note that if you change the number of input channels you cannot
    use pretrained models anymore.

    n_outputs (int): number of outputs of the last fully connected layer.
    n (str or int): number of layers of the resnet model. Existing versions:
        18, 34, 50, 101 and 151.
    """

    # Get resnet model from the dict of models and create an instance.
    if n not in _RESNETS:
        models_list = _RESNETS.items()
        raise Exception(f"Model resnet{n} not found, try with {models_list}")
    model = _RESNETS[str(n)](pretrained=pretrained)

    # Freeze the params if needed.
    if freeze:
        # It does not make sense to freeze a non-pretrained model.
        assert pretrained
        freeze(model)
        # Method for easily unfreeze model.
        model.unfreeze = lambda: unfreeze(model)

    # If the number of input channels is not 3 we need to recreate the first
    # conv layer. Here I'm using the same parameters except for the number
    # of input channels.
    if in_channels != 3:
        if pretrained:
            # TODO: change exception by warning like in efficientnets.
            raise Exception("If you change the number of input channels you "
                            "cannot use pretrained networks because you are "
                            "changing the first conv layer.")
        # TODO: Recreate conv using parameters of the existing one.
        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
    # Get the input number of features of the last linear layer.
    in_fc_features = model.fc.in_features
    # Create a new linear layer with the desired number of outputs.
    model.fc = torch.nn.Linear(in_fc_features, n_outputs)
    return model


def resnet18(n_outputs, **kwargs):
    return resnet(n_outputs, n="18", **kwargs)

