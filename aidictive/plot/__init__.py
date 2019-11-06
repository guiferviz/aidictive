
import pandas as pd

import numpy as np


class Plot(object):
    pass


def set_backend(backend_name="plotly"):
    global backend
    if backend_name == "plotly":
        from aidictive.plot import plotly
        backend = plotly
    else:
        raise ValueError(f"Unknown backend name: {backend_name}")


def scatter(x, y=None, **kwargs):
    if y is None:
        x = np.asarray(x)
        if len(x.shape) == 2 and x.shape[1] > 1:
            x, y = x[:, 0], x[:, 1]
        elif len(x.shape) == 1:
            x, y = x, np.zeros_like(x)
    return backend.scatter(x, y, **kwargs)


def line(x, y=None, **kwargs):
    if y is None:
        assert len(x.shape) == 2 and x.shape[1] > 1
        x, y = x[:, 0], x[:, 1]
    return backend.line(x, y, **kwargs)


def bar(x, y=None, **kwargs):
    if y is None:
        if type(x) == pd.Series:
            x, y = x.index.values, x.values
        else:
            assert len(x.shape) == 2 and x.shape[1] > 1
            x, y = x[:, 0], x[:, 1]
    return backend.bar(x, y, **kwargs)


def reduce(matrix, reducer="pca", **kwargs):
    x, y = None, None
    assert len(matrix.shape) == 2
    if matrix.shape[1] > 1:
        if matrix.shape[1] > 2:
            reducer = _get_reducer(reducer)
            matrix = reducer.fit_transform(matrix)
        x, y = matrix[:, 0], matrix[:, 1]
    elif matrix.shape[1] == 1:
        x, y = matrix[:, 0], np.zeros_like(matrix[:, 0])
    else:
        raise ValueError("Cannot plot the given matrix. Zero columns?")
    return x, y


def _get_reducer(reducer, n_components=2, **kwargs):
    """Return a dimensionality reduction object with SKLearn interface.

    If reducer is not an string we assume it is object that works
    as any SKLearn model.
    """
    if type(reducer) == str:
        if reducer.lower() in ["tsne", "t-sne"]:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, **kwargs)
        elif reducer.lower() == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components, **kwargs)
        elif reducer.lower() == "umap":
            from umap import UMAP
            reducer = UMAP(n_components=n_components, **kwargs)
        else:
            raise ValueError(f"Unknown reducer name '{reducer}'.")
    return reducer


def scatter_reduce(matrix, reducer="pca", **kwargs):
    x, y = reduce(matrix, reducer=reducer)
    return scatter(x, y, **kwargs)


backend = None
set_backend()

