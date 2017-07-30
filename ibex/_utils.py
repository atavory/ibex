import six
import numpy as np
import pandas as pd
from sklearn import base


def verify_x_type(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError('Expected pandas.DataFrame; got %s' % type(X))


def verify_y_type(y):
    if y is None:
        return

    if not isinstance(y, (pd.DataFrame, pd.Series)):
        raise TypeError('Expected pandas.DataFrame or pandas.Series; got %s' % type(y))


