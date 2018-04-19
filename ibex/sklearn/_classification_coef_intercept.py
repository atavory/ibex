from __future__ import absolute_import


import string

import pandas as pd

from .._base import _get_iris_example_doc_preamble_ as _get_iris_example_doc_preamble_


def coef_(self, base_ret):
    if len(base_ret.shape) == 1:
        return pd.Series(base_ret, index=self.x_columns)

    if len(base_ret.shape) == 2:
        if base_ret.shape[0] == 1:
            return pd.DataFrame(base_ret, columns=self.x_columns)
        index = self.classes_
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
        >>>
        >>> from ibex.sklearn import $orig as pd_$orig
        >>>
        >>> clf =  pd_$orig.$name($kwargs).fit(iris[features], iris['class'])
        >>>
        >>> clf.coef_
        sepal length (cm)   ...
        sepal width (cm)    ...
        petal length (cm)   ...
        petal width (cm)    ...
        dtype: float64

    """).substitute({
        'orig': orig, 'name': name,
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
        >>>
        >>> clf =  pd_$orig.$name($kwargs).fit(iris[features], iris[['class', 'class']])
        >>>
        >>> clf.coef_
                    sepal length (cm)  sepal width (cm)  petal length (cm)
        setosa              ...
        versicolor          ...
        virginica           ...
        <BLANKLINE>
                    petal width (cm)
        setosa             ...
        versicolor         ...
        virginica          ...
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
        indent=0) + \
    string.Template(
    r"""
        >>> from ibex.sklearn import $orig as pd_$orig
        >>>
        >>> clf = pd_$orig.$name($kwargs).fit(iris[features], iris['class'])
        >>>
        >>> clf.intercept_
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

        >>>
        >>> from ibex.sklearn import $orig as pd_$orig
        >>>
        >>> clf = pd_$orig.$name($kwargs).fit(iris[features], iris[['class', 'class']])
        >>>
        >>> clf.intercept_
        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
        0...
        1...
        2...

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
