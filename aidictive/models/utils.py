
def freeze(model, freeze=True):
    """Freeze all the parameters of the model.

    Freeze = not requires grad.
    """

    for param in model.parameters():
        param.requires_grad = not freeze


def unfreeze(model):
    """Unfreeze all the parameters of the model.

    Equivalent to `freeze(model, freeze=False)`.
    """

    freeze(model, False)

