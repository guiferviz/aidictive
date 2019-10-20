
import inspect

import pandas as pd

from . import plot
from .eda import memory_by_column, value_counts, unique_by_column, na_by_column
from .utils import reduce_memory_usage, top_others, default_params, \
    memory_usage_mb, compare_df_unique_values, compare_series_unique_values

SERIES_ACCESSOR_NAME = "eda"
DATAFRAME_ACCESSOR_NAME = "eda"


@pd.api.extensions.register_series_accessor(SERIES_ACCESSOR_NAME)
class EDASeriesAccessor(object):

    def __init__(self, s):
        self.s = s

    def na(self):
        """Get total number of nans/nulls. """
        return self.s.isna().sum()

    def compare_unique_values(self, *args, **kwargs):
        return compare_series_unique_values(self.s, *args, **kwargs)

    def value_counts(self, *args, **kwargs):
        return value_counts(self.s, *args, **kwargs)

    def plot_dist(self, **kwargs):
        args = {"bins": 50, "color": "b"}
        args.update(kwargs)
        return sns.distplot(self.s, **kwargs)

    def _repr_html_(self):
        html = "<div>EDA is very important! Explore this Series with:<ul>"
        methods = _get_eda_methods(EDASeriesAccessor)
        for i in methods:
            html += f"<li>{i}</li>"
        html += "</ul></div>"
        return html


@pd.api.extensions.register_dataframe_accessor(DATAFRAME_ACCESSOR_NAME)
class EDADataFrameAccessor(object):

    def __init__(self, df):
        self.df = df

    def memory_usage_mb(self, *args, **kwargs):
        return memory_usage_mb(self, *args, **kwargs)

    def reduce_memory_usage(self, *args, **kwargs):
        return reduce_memory_usage(self.df, *args, **kwargs)

    def memory_by_column(self, *args, **kwargs):
        return memory_by_column(self.df, *args, **kwargs)

    def unique_by_column(self, *args, **kwargs):
        return unique_by_column(self.df, *args, **kwargs)

    def compare_unique_values(self, *args, **kwargs):
        return compare_df_unique_values(self.df, *args, **kwargs)

    def is_pk_candidate(self, *args):
        df_unique = self.df.drop_duplicates(*args)
        return len(df_unique) == len(self.df)

    def na(self, relative=False):
        """Get total number of nans/nulls in all columns. """
        total = self.df.isna().sum().sum()
        if relative:
            total /= float(self.df.size)
        return total

    def na_by_column(self, *args, **kwargs):
        return na_by_column(self.df, *args, **kwargs)

    def plot_dist(self, column=None, **kwargs):
        if column is None:
            plots = []
            for i in self.df.columns:
                plots.append(self.df[i].eda.plot_dist())
        else:
            self.df[column].eda.plot_dist()

    def _repr_html_(self):
        html = "<div>EDA is very important! Explore this DataFrame using:<ul>"
        methods = _get_eda_methods(EDADataFrameAccessor)
        for i in methods:
            html += f"<li>{i}</li>"
        html += "</ul></div>"
        return html


def _get_eda_methods(accessor):
    methods = inspect.getmembers(accessor, predicate=inspect.isfunction)
    return [i[0] for i in methods if not i[0].startswith("_")]
