from __future__ import absolute_import


import functools
import inspect

import pandas as pd
from sklearn import base
from sklearn import cluster as orig

from .._adapter import frame_ex
from ._utils import get_matching_estimators


_extra_doc = """

# Tmp Ami

.. tip::

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.cluster import KMeans as PdKMeans

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

        >>> clt = PdKMeans(n_clusters=3, random_state=1).fit(iris[features])
        >>> clt.transform(iris[features])
            0         1         2
        0    3.419251  0.146942  5.059542
        1    3.398574  0.438169  5.114943
        2    3.569357  0.412301  5.279355
        3    3.422410  0.518837  5.153590
        4    3.467264  0.197970  5.104334
        ...

"""


def transform(self, base_ret):
    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = list(range(len(base_ret.columns)))
    return base_ret


def fit_transform(self, base_ret):
    if isinstance(base_ret, pd.DataFrame):
        base_ret.columns = list(range(len(base_ret.columns)))
    return base_ret


def update_module(module):
    module.__doc__ += _extra_doc

    for est in get_matching_estimators(module, base.TransformerMixin):
        est = frame_ex(
            getattr(orig, est.__name__),
            extra_methods=[transform, fit_transform])
        setattr(module, est.__name__, est)
