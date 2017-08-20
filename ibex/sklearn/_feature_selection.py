from __future__ import absolute_import


import functools
import inspect

import pandas as pd
from sklearn import base
from sklearn import feature_selection as orig

from .._adapter import frame_ex
from ._utils import get_matching_estimators


_extra_doc = """

.. tip::

    Transformers in this module label their columns by the original names they
    were called (prior to their selection).

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.feature_selection import SelectKBest as PdSelectKBest

        >>> iris = datasets.load_iris()
        >>> features = iris['feature_names']
        >>> iris = pd.DataFrame(
        ...     np.c_[iris['data'], iris['target']],
        ...     columns=features+['class'])

        >>> iris[features]
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0                5.1               3.5                1.4               0.2
        1                4.9               3.0                1.4               0.2
        2                4.7               3.2                1.3               0.2
        3                4.6               3.1                1.5               0.2
        4                5.0               3.6                1.4               0.2
        ...

        >>> PdSelectKBest(k=1).fit(iris[features], iris['class']).transform(iris[features])
            petal length (cm)
        0                  1.4
        1                  1.4
        2                  1.3
        3                  1.5
        4                  1.4
        ...

"""


def transform(self, base_ret):
    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = self.x_columns[self.get_support(indices=True)]
    return base_ret


def fit_transform(self, base_ret):
    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = self.x_columns[self.get_support(indices=True)]
    return base_ret


def update_module(module):
    module.__doc__ += _extra_doc

    for est in get_matching_estimators(module, base.TransformerMixin):
        if not hasattr(est, 'get_support'):
            continue
        est = frame_ex(
            getattr(orig, est.__name__),
            extra_methods=[transform, fit_transform])
        setattr(module, est.__name__, est)





