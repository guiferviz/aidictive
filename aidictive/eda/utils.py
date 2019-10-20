
import numpy as np

import pandas as pd


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

