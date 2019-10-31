
import numpy as np

from aidictive import plot
from aidictive import utils


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
    kwargs = utils.default_params(kwargs, x_cmap="tab10")
    p = plot.bar(c, **kwargs)
    return p(title="Count by value")


def unique_by_column(df, **kwargs):
    s = df.nunique().sort_values(ascending=False)
    n_rows = df.shape[0]
    title = f"# Unique values by Column (df # rows: {n_rows})"
    kwargs = utils.default_params(kwargs, title=title)
    relative = kwargs.pop("relative", None)
    relative = len( df) if relative is not None else relative
    return _prepare_and_plot_series(s, relative=relative, **kwargs)


def memory_by_column(df, deep=True, **kwargs):
    # Sorted pandas series with the MB used by each column.
    s = (df.memory_usage(deep=deep) / 1024**2).sort_values(ascending=False)
    kwargs = utils.default_params(kwargs, title="Memory Usage (MB) by Column")
    return _prepare_and_plot_series(s, **kwargs)


def na_by_column(df, **kwargs):
    # Sorted pandas series with the number of nans/nulls of each column.
    na = df.isna().sum().sort_values(ascending=False)
    kwargs = utils.default_params(kwargs, title="# NAs by Column")
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
    kwargs = utils.default_params(kwargs, x_cmap="tab10")
    p = plot.bar(s, **kwargs)
    return p(title=title)


def top_others(s, top, others="(others)", agg_fun="sum"):
    """Show the top rows by value and aggregate the others.

    Args:
        s (pandas.Series): Series from which to obtain the top rows.
        top (int): The second parameter. Defaults to None.
            Second line of description should be indented.
        others (str or any other value): `str` with the
            name of the others category. Any other value to
            ignore the rows that are not on the top.
            Default: `"(others)"`.
        agg_fun (str or function): Function used to
            aggregate the rows that are not on top and show
            that value as others category. You can use any
            value that `pd.Series.agg` accepts.
            Default: `"sum"`.

    Returns:
        pandas.Series: A serie with the top rows order by value.
            If `top` is specified, the rest of the values are
            ignored if `other` is not an string. If `other` is
            an string, the non-top rows are aggregated using
            `agg_fun`.
    """
    s_top = None
    if top > 0:
        s_top = s.nlargest(top, keep="first")
    elif top < 0:
        s_top = s.nsmallest(-top, keep="first")
    else:
        raise ValueError("top cannot be zero")
    if type(others) == str:
        n_others = s.drop(s_top.index).agg(agg_fun)
        s_top = s_top.append(pd.Series([n_others], index=[others]))
    return s_top


def get_unique_values_serie(s):
    """Get an array with the unique values of s.

    Args:
        s (pandas.Series): Series from which you want to obtain the unique
            values.

    This function takes advantage of category types. The array
    `s.cat.categories` is indeed an unique vector.
    """
    if s.dtype.name == "category":
        return s.cat.categories
    return np.unique(s.values)


def compare_series_unique_values(s1, s2):
    a, b = get_unique_values_serie(s1), get_unique_values_serie(s2)
    intersection = np.intersect1d(a, b, assume_unique=True)
    return len(a), len(intersection), len(b)


def compare_df_unique_values(df1, df2):
    results = []
    columns = df1.columns.union(df2.columns)
    for col in columns:
        print(f"Evaluating column {col}...")
        if col in df1 and col in df2:
            la, iab, lb = compare_series_unique_values(df1[col], df2[col])
            results.append([col, la, iab, lb])
        elif col in df1:
            u = get_unique_values_serie(df1[col])
            results.append([col, len(u), -1, -1])
        elif col in df2:
            u = get_unique_values_serie(df1[col])
            results.append([col, -1, -1, len(u)])
    results = pd.DataFrame(results)
    results.columns = ["column_name", "# unique in df1",
                       "# unique in df1 and df2", "# unique in df2"]
    results.set_index("column_name", inplace=True)
    return results


def is_pk_candidate(df, *columns):
    df_unique = df.drop_duplicates(*columns)
    return len(df_unique) == len(df)

