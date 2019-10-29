
import inspect

import pandas as pd

from . import plot
from . import eda
from . import utils


SERIES_ACCESSOR_NAME = "aidictive"
DATAFRAME_ACCESSOR_NAME = "aidictive"


@pd.api.extensions.register_series_accessor(SERIES_ACCESSOR_NAME)
class AIdictiveSeriesAccessor(object):
    """Pandas Serie accessor with AIdictive methods. """

    def __init__(self, s):
        self.s = s

    def na(self):
        """Get total number of nans/nulls. """

        return self.s.isna().sum()

    def memory_usage_mb(self, *args, **kwargs):
        """Get the memory usage of this serie in MB. """

        return utils.memory_usage_mb(self.s, *args, **kwargs)

    def compare_unique_values(self, *args, **kwargs):
        return eda.compare_series_unique_values(self.s, *args, **kwargs)

    def value_counts(self, *args, **kwargs):
        """See `aidictive.eda.value_counts`. """

        return eda.value_counts(self.s, *args, **kwargs)

    def _repr_html_(self):
        html = "<div>Explore this Series with:<ul>"
        methods = _get_eda_methods(AIdictiveSeriesAccessor)
        for i in methods:
            html += f"<li>{i}</li>"
        html += "</ul></div>"
        return html


@pd.api.extensions.register_dataframe_accessor(DATAFRAME_ACCESSOR_NAME)
class AIdictiveDataFrameAccessor(object):
    """Pandas DataFrame accessor with AIdictive methods. """

    def __init__(self, df):
        self.df = df

    def memory_usage_mb(self, *args, **kwargs):
        """Get the memory usage of all the dataframe in MB.
        
        It performs a call to the `pandas.DataFrame.memory_usage`, so you
        can add all the parameters that that function accepts, like
        `deep=True`.
        """

        return utils.memory_usage_mb(self.df, *args, **kwargs)

    def reduce_memory_usage(self, *args, **kwargs):
        """Call to `aidictive.utils.reduce_memory_usage`. """

        return utils.reduce_memory_usage(self.df, *args, **kwargs)

    def memory_by_column(self, *args, **kwargs):
        return memory_by_column(self.df, *args, **kwargs)

    def unique_by_column(self, *args, **kwargs):
        return unique_by_column(self.df, *args, **kwargs)

    def compare_unique_values(self, *args, **kwargs):
        return compare_df_unique_values(self.df, *args, **kwargs)

    def is_pk_candidate(self, *args):
        return eda.is_pk_candidate(self.df, *args)

    def is_unique(self, *args):
        return eda.is_pk_candidate(self.df, *args)

    def na(self, relative=False):
        """Get total number of nans/nulls in all columns. """
        total = self.df.isna().sum().sum()
        if relative:
            total /= float(self.df.size)
        return total

    def na_by_column(self, *args, **kwargs):
        return eda.na_by_column(self.df, *args, **kwargs)

    def nan_by_column(self, *args, **kwargs):
        return eda.na_by_column(self.df, *args, **kwargs)

    def nans_by_column(self, *args, **kwargs):
        return eda.na_by_column(self.df, *args, **kwargs)

    def null_by_column(self, *args, **kwargs):
        return eda.na_by_column(self.df, *args, **kwargs)

    def nulls_by_column(self, *args, **kwargs):
        return eda.na_by_column(self.df, *args, **kwargs)

    def plot_dist(self, column=None, **kwargs):
        if column is None:
            plots = []
            for i in self.df.columns:
                plots.append(self.df[i].eda.plot_dist())
        else:
            self.df[column].eda.plot_dist()

    def _repr_html_(self):
        html = "<div>Explore this DataFrame using:<ul>"
        methods = _get_eda_methods(AIdictiveDataFrameAccessor)
        for i in methods:
            html += f"<li>{i}</li>"
        html += "</ul></div>"
        return html


def _get_eda_methods(accessor):
    methods = inspect.getmembers(accessor, predicate=inspect.isfunction)
    return [i[0] for i in methods if not i[0].startswith("_")]

