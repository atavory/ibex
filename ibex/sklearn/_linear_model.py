from __future__ import absolute_import


import inspect

import pandas as pd
from sklearn import base
from sklearn import linear_model as orig

from .._adapter import frame_ex
from ._utils import get_matching_estimators


_extra_doc = """

.. tip::

    Transformers in this module label their columns as ``comp_0``, ``comp_1``, and so on.

    Example

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.decomposition import PCA as PDPCA

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

        >>> PDPCA(n_components=2).fit(iris[features], iris['class']).transform(iris[features])
            comp_0    comp_1
        0   -2.684207  0.326607
        1   -2.715391 -0.169557
        2   -2.889820 -0.137346
        3   -2.746437 -0.311124
        4   -2.728593  0.333925
        ...

"""


def coef_(self, base_ret):
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.y_columns if self.y_columns is not None else self.classes_
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError


def intercept_(self, base_ret):
    ggg
    if self.y_columns is not None:
        for _ in range(20):
            print(base_ret, self.x_columns, self.y_columns)
    return base_ret


def update_module(module):
    module.__doc__ += _extra_doc

    for est in get_matching_estimators(module, base.BaseEstimator):
        est = frame_ex(
            getattr(orig, est.__name__),
            extra_attribs=[coef_, intercept_])
        setattr(module, est.__name__, est)

    # Tmp Ami
    # getattr(module, 'LinearRegression')().coef_





