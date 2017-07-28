from __future__ import absolute_import


import operator

import six

from ibex.sklearn import pipeline as _pd_pipeline


def make_pipeline(*estimators):
    """
    Creates a pipeline from estimators.

    Arguments:

        transformers: Iterable of estimators.

    Returns:

        A :class:`ibex.sklearn.pipeline.Pipeline` object.

    Example:

        >>> from ibex.sklearn import preprocessing
        >>> from ibex.sklearn import linear_model
        >>> from ibex.sklearn import pipeline
        >>>
        >>> pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.LinearRegression())
        Pipeline(steps=[...]...)
    """
    estimators = list(estimators)

    if len(estimators) > 1:
        return six.moves.reduce(operator.or_, estimators[1: ], estimators[0])

    name = type(estimators[0]).__name__.lower()
    return _pd_pipeline.Pipeline([(name, estimators[0])])


def make_union(*transformers):
    """
    Creates a union from transformers.

    Arguments:

        transformers: Iterable of transformers.

    Returns:

        A :class:`ibex.sklearn.pipeline.FeatureUnion` object.

    Example:

        >>> from ibex.sklearn import preprocessing as pd_preprocessing
        >>> from ibex.sklearn import pipeline as pd_pipeline

        >>> trn = pd_pipeline.make_union(pd_preprocessing.StandardScaler(), pd_preprocessing.MaxAbsScaler())
    """

    transformers = list(transformers)
    return six.moves.reduce(operator.add, transformers[1: ], transformers[0])
