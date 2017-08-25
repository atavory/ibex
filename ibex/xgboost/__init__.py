"""
Wrappers for :mod:`xgboost`.


.. versionadded:: 1.2


Example:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from ibex.sklearn import datasets
    >>> from ibex.xgboost import XGBRegressor as PdXGBRegressor
    >>> from ibex.xgboost import XGBClassifier as PdXGBClassifier

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

    >>> PdXGBRegressor().fit(iris[features], iris['class']).predict(iris[features])
    0      0.000140
    1      0.013568
    2     -0.004346
    3      0.005156
    4      0.000140
    ...

    >>> PdXGBClassifier().fit(iris[features], iris['class']).predict(iris[features])
    0      0...
    1      0...
    2      0...
    3      0...
    4      0...
    ...

"""


from __future__ import absolute_import

import xgboost

from .._adapter import frame


XGBClassifier = frame(xgboost.XGBClassifier)
XGBRegressor = frame(xgboost.XGBRegressor)


__all__ = ['XGBClassifier', 'XGBRegressor']




