
import torch

from aidictive.activations import get as get_activation


class FFNN(torch.nn.Module):
    """N layers NN for regression. """

    def __init__(self, layers, activations="selu", last_activation="same"):
        super().__init__()
        if type(activations) == str:
            activations = ["linear"] + [activations] * (len(layers) - 2)
        if last_activation == "same":
            last_activation = activations[-1]
        activations += [last_activation]
        assert len(activations) == len(layers)
        modules = []
        for i in range(1, len(layers)):
            layer = torch.nn.Linear(layers[i - 1], layers[i])
            modules.append(layer)
            activation = get_activation(activations[i])
            if activation is not None:
                modules.append(activation)
        self.seq = torch.nn.Sequential(*modules)

    def forward(self, X):
        return self.seq(X)

