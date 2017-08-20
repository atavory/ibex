from __future__ import absolute_import


import inspect

import pandas as pd
from sklearn import base
from sklearn import ensemble as orig

from .._adapter import frame_ex
from ._utils import get_matching_estimators


_extra_doc = """

.. tip::

    Estimators in this module have a ``feature_importances_`` attribute following a
    call to ``fit*``.

    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.ensemble import RandomForestClassifier as PDRandomForestClassifier

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

        >>> clf =  PDRandomForestClassifier(random_state=42).fit(iris[features], iris['class'])
        >>>
        >>> clf.feature_importances_
        sepal length (cm)    0.129268
        sepal width (cm)     0.015822
        petal length (cm)    0.444740
        petal width (cm)     0.410169
        dtype: float64

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PDLogisticRegression

        >>> clf =  PDLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0           0.414988          1.461297          -2.262141         -1.029095
        1           0.416640         -1.600833           0.577658         -1.385538
        2          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

"""


def feature_importances_(self, base_ret):
    return pd.Series(base_ret, index=self.x_columns)


def update_module(module):
    module.__doc__ += _extra_doc

    for est in get_matching_estimators(module, base.BaseEstimator):
        est = frame_ex(
            getattr(orig, est.__name__),
            extra_attribs=[feature_importances_])
        setattr(module, est.__name__, est)





