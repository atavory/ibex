from __future__ import absolute_import


import pandas as pd


__all__ = []


def verify_x_type(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('Expected pandas.DataFrame; got %s' % type(X))

__all__ += ['verify_x_type']


def verify_y_type(y):
    if y is None:
        return

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError('Expected pandas.DataFrame or pandas.Series; got %s' % type(y))

__all__ += ['verify_y_type']
