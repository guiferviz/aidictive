
import sklearn


_METRICS = {
    "mse": sklearn.metrics.mean_squared_error,
    "mae": sklearn.metrics.mean_absolute_error,
    "mean_squared_error": sklearn.metrics.mean_squared_error,
    "accuracy": sklearn.metrics.accuracy_score,
    "auc": sklearn.metrics.auc,
    "f1": sklearn.metrics.f1_score,
    "f1_score": sklearn.metrics.f1_score,
    "precision": sklearn.metrics.precision_score,
    "recall": sklearn.metrics.recall_score,
}


def add(name, fun):
    """Add a new metric to the library. """

    _METRICS[name] = fun


def get(name_or_fun):
    """Get a metric function or return the given object. """

    if type(name_or_fun) == str:
        return _METRICS[name_or_fun]
    return name_or_fun

