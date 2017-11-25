from __future__ import absolute_import


import pandas as pd


def feature_importances_(self, base_ret):
    return pd.Series(base_ret, index=self.x_columns)


def get_feature_importances_docs(
        orig,
        name,
        est,
        kwargs,
        is_regressor,
        is_classifier,
        is_transformer,
        is_clusterer,
        has_dataframe_y):
    return r"""
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.ensemble import RandomForestClassifier as PdRandomForestClassifier

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

        >>> clf =  PdRandomForestClassifier(random_state=42).fit(iris[features], iris['class'])
        >>>
        >>> clf.feature_importances_
        sepal length (cm)    0.129268
        sepal width (cm)     0.015822
        petal length (cm)    0.444740
        petal width (cm)     0.410169
        dtype: float64

    """


