
import torch
import torchvision


_RESNETS = {
    "18": torchvision.models.resnet18,
    "34": torchvision.models.resnet34,
    "50": torchvision.models.resnet50,
    "101": torchvision.models.resnet101,
    "152": torchvision.models.resnet152,
}


def freeze(model, freeze=True):
    for param in model.parameters():
        param.requires_grad = not freeze


def resnet(n_outputs, n="18", input_channels=3, pretrained=False):
    """Create a resnet model using torchvision.

    You can ajust the number of input_channels and the number of outputs.
    Please, note that if you change the number of input channels you cannot
    use pretrained models anymore.

    n_outputs (int): number of outputs of the last fully connected layer.
    n (str or int): number of layers of the resnet model. Existing versions:
        18, 34, 50, 101 and 151.
    """

    # Get resnet model from the dict of models.
    if n not in _RESNETS:
        models_list = _RESNETS.items()
        raise Exception(f"Model resnet{n} not found, try with {models_list}")
    model = _RESNETS[str(n)](pretrained=pretrained)
    # If the number of input channels is not 3 we need to recreate the first
    # conv layer. Here I'm using the same parameters except for the number
    # of input channels.
    if input_channels != 3:
        if pretrained:
            raise Exception("If you change the number of input channels you "
                            "cannot use pretrained networks because you are "
                            "changing the first conv layer.")
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7,
                                stride=2, padding=3, bias=False)
    # Freeze the params if needed.
    if pretrained:
        freeze(model)
    # Get the input number of features of the last linear layer.
    num_ftrs = model.fc.in_features
    # Create a new linear layer with the desired number of outputs.
    model.fc = torch.nn.Linear(num_ftrs, n_outputs)
    return model


def resnet18(n_outputs, **kwargs):
    return resnet(n_outputs, n="18", **kwargs)

