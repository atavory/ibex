from __future__ import absolute_import


import pandas as pd


def coef_(self, base_ret):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

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

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.classes_
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError()


def get_coef_doc(
        orig, name, est, kwargs, is_classifier, is_transformer, is_clusterer):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

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

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """


def intercept_(self, base_ret):

    # Tmp Ami - replace next by is_nummeric or is_scalar
    if isinstance(base_ret, (type(1), type(1.), type(1 + 1j))):
        return base_ret

    if len(base_ret.shape) == 1:
        return pd.Series(base_ret)

    raise RuntimeError()


def get_intercept_doc(
        orig, name, est, kwargs, is_classifier, is_transformer, is_clusterer):
    """
    Example:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from ibex.sklearn import datasets
        >>> from ibex.sklearn.linear_model import LinearRegression as PdLinearRegression

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

        >>> prd =  PdLinearRegression().fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   -0.109741
        sepal width (cm)    -0.044240
        petal length (cm)    0.227001
        petal width (cm)     0.609894
        dtype: float64
        >>>
        >>> prd.intercept_
        0.19208...

    Example:

        >>> from ibex.sklearn.linear_model import LogisticRegression as PdLogisticRegression

        >>> clf =  PdLogisticRegression().fit(iris[features], iris['class'])
        >>> clf.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382
        >>> clf.intercept_
        0    0.265606
        1    1.085424
        2   -1.214715
        dtype: float64

    """

