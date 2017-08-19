import functools

import six
import pandas as pd
import numpy as np


_wrap_msg = """

.. note::

    The documentation following is of the class wrapped by this class. There
    are some changes, in particular:

        * A parameter ``X`` denotes a :class:`pandas.DataFrame`.

        * A parameter ``y`` denotes a :class:`pandas.Series`.

--------------------------------

"""


def verify_x_type(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('Expected pandas.DataFrame; got %s' % type(X))


def verify_y_type(y):
    if y is None:
        return

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError('Expected pandas.DataFrame or pandas.Series; got %s' % type(y))


def update_class_wrapper(new_class, orig_class):
    if six.PY3:
        orig_doc = orig_class.__doc__
        if orig_doc is None:
            orig_doc = ''
        new_class.__doc__ = _wrap_msg + orig_doc


def update_method_wrapper(new_class, orig_class, method_name):
    functools.update_wrapper(getattr(new_class, method_name), getattr(orig_class, method_name))
    orig_doc = getattr(orig_class, method_name).__doc__
    if orig_doc is None:
        orig_doc = ''
    getattr(new_class, method_name).__doc__ = _wrap_msg + orig_doc


_have_y_fn_names = [
    'fit',
    'fit_predict',
    'fit_transform',
    'inverse_transform',
    'partial_fit',
    'score',
    'transform',
]


def get_wrapped_y(name, args):
    if name not in _have_y_fn_names:
        return None

    if len(args) < 1:
        return None

    if isinstance(args[0], (pd.Series, pd.DataFrame, np.ndarray)):
        return args[0]

    return None


def update_wrapped_y(args, y):
    args[0] = y


wrapped_fn_names = [
    'fit_transform',
    'predict_proba',
    'sample_y',
    'score_samples',
    'score',
    'staged_predict_proba',
    'apply',
    'bic',
    'perplexity',
    'fit',
    'decision_function',
    'aic',
    'partial_fit',
    'predict',
    'radius_neighbors',
    'staged_decision_function',
    'staged_predict',
    'inverse_transform',
    'fit_predict',
    'kneighbors',
    'predict_log_proba',
    'transform',
]
