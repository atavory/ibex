import functools

import six
import pandas as pd


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
        new_class.__doc__ = _wrap_msg + orig_class.__doc__


def update_method_wrapper(new_class, orig_class, method_name):
    functools.update_wrapper(getattr(new_class, method_name), getattr(orig_class, method_name))
    getattr(new_class, method_name).__doc__ = _wrap_msg + getattr(orig_class, method_name).__doc__
