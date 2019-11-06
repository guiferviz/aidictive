
import numpy as np

import pandas as pd

import sklearn.model_selection

import torch

from aidictive.data.datasets import mnist, mnist_datasets, mnist_data_loaders


def get_tensor_data_loader(*args, **kwargs):
    """Create a data loader from numpy arrays, pytorch tensors or datasets.
 
    All args should be of the same type: or numpy arrays or pytorch tensors.
    Any extra named parameter is going to be used to create the DataLoader.
    """

    # At least one array, tensor or dataset!
    assert len(args) > 0
    # Check if the given arg is a dataset.
    if isinstance(args[0], torch.utils.data.dataset.Dataset):
        ds = args[0]
    else:
        # Check that all the types are the same in args.
        type_data = type(args[0])
        assert all(type(i) == type_data or i is None for i in args)

        if type_data == np.ndarray:
            # Converting numpy arrays to tensors.
            args = [torch.tensor(i) for i in args]
        ds = torch.utils.data.dataset.TensorDataset(*args)
    dl = torch.utils.data.DataLoader(ds, **kwargs)
    return dl


def split(*args, group=None, test_size=0.3, random_state=None, shuffle=None,
          stratify=None):
    """Utility method that wraps several sklearn split methods. """

    # At least some data to split.
    assert len(args) > 0

    if group is None:
        # Random split.
        train, test = sklearn.model_selection.train_test_split(*args,
                test_size=test_size, random_state=random_state,
                shuffle=shuffle, stratify=stratify)
        return train, test

    # Group split.
    # If group split we assume that you are only giving one dataframe in args.
    assert len(args) == 1
    df = args[0]
    assert type(df) == pd.DataFrame
    # Check that you are not using unuseful parameters for this kind of split.
    assert shuffle is None and stratify is None
    # If group is a string it should be the name of a column in the df
    if type(group) == str:
        group = df[group]
    # Create splits.
    splitter = sklearn.model_selection.GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=group))
    train, test = df.iloc[train_idx, :], df.iloc[test_idx, :]
    return train, test

