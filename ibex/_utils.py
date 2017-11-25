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


# Tmp Ami - use this
_predict_log_star_doc = """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression
        >>>
        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> clf = PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.classes_
        array([ 0.,  1.,  2.])

        >>> clf.predict_proba(iris[features])
                0...      1...          2...
        0    0.879682  0.120308  1.081314e-05
        1    0.799706  0.200263  3.038254e-05
        2    0.853797  0.146177  2.590313e-05
        3    0.825383  0.174559  5.793567e-05
        4    0.897324  0.102665  1.120500e-05
        5    0.926987  0.073000  1.296939e-05
        ...

        >>> clf.predict_log_proba(iris[features])
                0...      1...       2...
        0    -0.128195 -2.117704 -11.434749
        1    -0.223511 -1.608122 -10.401643
        2    -0.158062 -1.922935 -10.561147
        3    -0.191908 -1.745493  -9.756177
        4    -0.108339 -2.276282 -11.399150
        5    -0.075816 -2.617290 -11.252919
        ...

"""
