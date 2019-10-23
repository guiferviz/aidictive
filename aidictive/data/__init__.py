
import numpy as np

import torch


def get_tensor_data_loader(*args, **kwargs):
    """Create a data loader from numpy arrays or pytorch tensors.
 
    All args should be of the same type: or numpy arrays or pytorch tensors.
    Any extra named parameter is going to be used to create the DataLoader.
    """

    # At least one array or tensor!
    assert len(args) > 0
    # Check that all the types are the same in args.
    type_data = type(args[0])
    assert all(type(i) == type_data for i in args)

    if type_data == np.ndarray:
        # Converting numpy arrays to tensors.
        args = [torch.tensor(i) for i in args]
    ds = torch.utils.data.dataset.TensorDataset(*args)
    dl = torch.utils.data.DataLoader(ds, **kwargs)
    return dl

