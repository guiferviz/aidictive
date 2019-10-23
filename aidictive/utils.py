
import fnmatch


def default_params(fun_kwargs, default_dict=None, **kwargs):
    """Add to kwargs and/or default_dict the values of fun_kwargs. """
    if default_dict is None:
        default_dict = kwargs
    else:
        default_dict.update(kwargs)
    default_dict.update(fun_kwargs)
    return default_dict


def columns_match(df, cols, error=True):
    cols_match = []
    for i in cols:
        cols_match_aux = fnmatch.filter(list(df.columns), i)
        if error and len(cols_match_aux) == 0:
            raise ValueError(f"No column match '{i}' in dataframe")
        cols_match += cols_match_aux
    return cols_match


def reduce_memory_usage(df, deep=True, verbose=True, categories=True):
    """Change datatypes in order to reduce memory usage.

    This function iterates all the columns of the dataframe and
    for numeric types checks if there is another datatype that
    occupies less memory and can store all the numbers of the
    column.

    Another way to reduce memory usage of Pandas dataframes is
    creating categories. This function in verbose mode shows
    if there are columns of type object that could be converted
    to category. It is not always possible to save memory by
    converting to a category. The ideal columns to convert to
    category are those with few different values. Converting a
    column that has all unique values will surely occupy more as
    a category.
    In a dataframe there may also be numeric columns that can be
    converted to categorical (especially columns of integer
    types). Category conversions must be done by the user.

    Args:
        df (pandas.DataFrame): Dataframe to change data types.
    """
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type == "object":
            df[col] = df[col].astype("category")
            best_type = "category"
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")
    return df


def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024**2


def init_categories(df, category_cols, ordered=None, ignore_error_keys=False):
    """Transform columns of a dataframe into categories.

    Args:
        ignore_error_keys (bool): If `True`, any column of `category_cols`
            not contained in `df` will be ignored.
    """
    if ordered is None:
        ordered = [False] * len(category_cols)
    for i, o in zip(category_cols, ordered):
        if not ignore_error_keys or i in df:
            df[[i]] = df[[i]].apply(lambda col: pd.Categorical(col,
                # WTF means the next conditional?? What I was thinking about?
                ordered=col.dtype == int if o is None else o))
    return df


def common_categories(df_origin, df_destination):
    """Apply the categories in df_origin to df_destination.

    The dataframes can be different, this method only looks
    for columns of the same name as in df_destination in df_origin
    and transform them to category if that columns are categories
    in df_origin.
    The category levels and the order of the category will be
    the same for columns with the same name after applying this
    method.
    """
    for i in df_destination:
        if i in df_origin and df_origin[i].dtype.name == "category":
            df_destination[i] = pd.Categorical(
                df_destination[i],
                categories=df_origin[i].cat.categories,
                ordered=df_origin[i].cat.ordered)

    # TODO: Apply also to index (?) Not sure if we can do this.
    return df_destination


def is_jupyter_notebook():
    """Check if Python is running on a Jupyter Notebook.

    Thanks to: https://stackoverflow.com/a/39662359/5114276
    Personal addition: Google Colab detection as a notebook.
    """
    try:
        shell = get_ipython().__class__
        # Jupyter notebook or qtconsole.
        if shell.__name__ == "ZMQInteractiveShell":
            return True, "notebook"
        # Terminal running IPython.
        elif shell.__name__ == "TerminalInteractiveShell":
            return False, "ipython_terminal"
        # Google Colab notebook.
        elif shell.__module__ == "google.colab._shell":
            return True, "google_colab"
        # Other type (?).
        else:
            return False, "other?"
    except NameError:
        # If get_ipython() is not defined we are probably
        # in standard Python.
        return False, "standard_python?"
