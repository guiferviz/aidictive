
def truncated_normal_(tensor, mean=0.0, std=0.01):
    """Truncated normal initialization method.

    Taken from:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

