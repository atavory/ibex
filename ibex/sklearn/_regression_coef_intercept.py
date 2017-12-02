from __future__ import absolute_import


import string

import pandas as pd

from .._base import _get_iris_example_doc_preamble_ as _get_iris_example_doc_preamble_


def coef_(self, base_ret):
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        index = self.y_columns
        return pd.DataFrame(base_ret, index=index, columns=self.x_columns)

    raise RuntimeError()


def get_coef_doc(
        orig,
        name,
        est,
        kwargs,
        is_regressor,
        is_classifier,
        is_transformer,
        is_clusterer,
        has_dataframe_y):

    doc = _get_iris_example_doc_preamble_(
        is_regressor,
        is_classifier,
        is_transformer,
        is_clusterer,
        indent=0) + \
    string.Template(
    r"""
        >>> from ibex.sklearn import $orig as pd_$orig
        >>>
        >>> prd =  pd_$orig.$name($kwargs).fit(iris[features], iris['class'])
        >>>
        >>> prd.coef_
        sepal length (cm)   ...
        sepal width (cm)    ...
        petal length (cm)   ...
        petal width (cm)    ...
        dtype: float64

    """).substitute({
        'orig': orig,
        'name': name,
        'est': est,
        'kwargs': kwargs,
        'is_regressor': is_regressor,
        'is_classifier': is_classifier,
        'is_transformer': is_transformer,
        'is_clusterer': is_clusterer})


    if has_dataframe_y:
        doc += string.Template(
    r"""

    Example:

        >>> from ibex.sklearn import $orig as pd_$orig
        >>> prd =  pd_$orig.$name($kwargs).fit(iris[features], iris[['class', 'class']])
        >>>
        >>> prd.coef_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...           0.414988          1.461297          -2.262141         -1.029095
        1...           0.416640         -1.600833           0.577658         -1.385538
        2...          -1.707525         -1.534268           2.470972          2.555382

    """).substitute({
        'orig': orig,
        'name': name,
        'est': est,
        'kwargs': kwargs,
        'is_regressor': is_regressor,
        'is_classifier': is_classifier,
        'is_transformer': is_transformer,
        'is_clusterer': is_clusterer})

    return doc


def intercept_(self, base_ret):
    # Tmp Ami - replace next by is_nummeric or is_scalar
    if isinstance(base_ret, (type(1), type(1.), type(1 + 1j))):
        return base_ret

    if len(base_ret.shape) == 1:
        return pd.Series(base_ret)

    raise RuntimeError()


def get_intercept_doc(
        orig,
        name,
        est,
        kwargs,
        is_regressor,
        is_classifier,
        is_transformer,
        is_clusterer,
        has_dataframe_y):

    doc = _get_iris_example_doc_preamble_(
        is_regressor,
        is_classifier,
        is_transformer,
        is_clusterer,
        indent=0)

    if not has_dataframe_y:
        doc += string.Template(
    r"""
        >>>
        >>> from ibex.sklearn import $orig as pd_$orig
        >>>
        >>> prd = pd_$orig.$name($kwargs).fit(iris[features], iris['class'])
        >>>
        >>> #scalar intercept
        >>> type(prd.intercept_)
        <class 'numpy.float64'>

    """).substitute({
        'orig': orig,
        'name': name,
        'est': est,
        'kwargs': kwargs,
        'is_regressor': is_regressor,
        'is_classifier': is_classifier,
        'is_transformer': is_transformer,
        'is_clusterer': is_clusterer})


    if has_dataframe_y:
        doc += string.Template(
    r"""

        >>> from ibex.sklearn import $orig as pd_$orig
        >>> prd = pd_$orig.$name($kwargs).fit(iris[features], iris[['class', 'class']])
        >>>
        >>> prd.intercept_
        sepal length (cm)   ...
        sepal width (cm)    ...
        petal length (cm)   ...
        petal width (cm)    ...
        dtype: float64

    """).substitute({
        'orig': orig,
        'name': name,
        'est': est,
        'kwargs': kwargs,
        'is_regressor': is_regressor,
        'is_classifier': is_classifier,
        'is_transformer': is_transformer,
        'is_clusterer': is_clusterer})

    return doc
