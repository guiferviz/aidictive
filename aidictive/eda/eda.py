
import numpy as np

from . import plot
from .utils import default_params, top_others


def value_counts(s, dropna=False, top=None, relative=False,
                 others="(others)", nulls="(nulls)",
                 return_df=False, **kwargs):
    c = s.value_counts(dropna=dropna)
    c = c.rename(index={np.nan: nulls})
    if top is not None:
        c = top_others(c, top, others=others)
    if relative:
        c /= c.sum()
    if return_df:
        return c
    kwargs = default_params(kwargs, x_cmap="tab10")
    p = plot.bar(c, **kwargs)
    return p(title="Count by value")


def unique_by_column(df, **kwargs):
    s = df.nunique().sort_values(ascending=False)
    n_rows = df.shape[0]
    title = f"# Unique values by Column (df # rows: {n_rows})"
    kwargs = default_params(kwargs, title=title)
    relative = kwargs.pop("relative", None)
    relative = len( df) if relative is not None else relative
    return _prepare_and_plot_series(s, relative=relative, **kwargs)


def memory_by_column(df, deep=True, **kwargs):
    # Sorted pandas series with the MB used by each column.
    s = (df.memory_usage(deep=deep) / 1024**2).sort_values(ascending=False)
    kwargs = default_params(kwargs, title="Memory Usage (MB) by Column")
    return _prepare_and_plot_series(s, **kwargs)


def na_by_column(df, **kwargs):
    # Sorted pandas series with the number of nans/nulls of each column.
    na = df.isna().sum().sort_values(ascending=False)
    kwargs = default_params(kwargs, title="# NAs by Column")
    relative = kwargs.pop("relative", None)
    relative = len(df) if relative is not None else relative
    return _prepare_and_plot_series(na, relative=relative, **kwargs)


def _prepare_and_plot_series(s, top=None, others="(others)", relative=None,
                             title="Best plot ever :)", **kwargs):
    if top is not None:
        s = top_others(s, top, others=others)
    if relative is not None:
        relative = "sum" if relative is True else relative
        if type(relative) == str:
            total = s.agg(relative)
        elif type(relative) in [float, int]:
            total = relative
        else:
            raise ValueError(f"Unknown value for relative: f{relative}")
        s /= total
    kwargs = default_params(kwargs, x_cmap="tab10")
    p = plot.bar(s, **kwargs)
    return p(title=title)
