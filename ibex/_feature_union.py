from __future__ import absolute_import

import functools

import pandas as pd
from sklearn import base
from sklearn import pipeline
from sklearn.externals import joblib

from ._frame_mixin import FrameMixin



def _fit(transformer, name, weight, X):
    res = transformer.transform(X)
    return res if weight is None else weight * res


def _transform(transformer, name, weight, X):
    res = transformer.transform(X)
    return res if weight is None else weight * res


def _fit_transform(transformer, name, weight, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = transformer.fit_transform(X, y, **fit_params)
    else:
        res = transformer.fit(X, y, **fit_params).transform(X)
    return res if weight is None else weight * res


# Tmp Ami - take care of weights
# Tmp Ami - derive from appropriate sklearn mixins
class _FeatureUnion(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    """
    - Pandas version -
    Concatenates results of multiple transformer objects.
    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Arguments:

        transformer_list: list of (string, transformer) tuples.
            List of transformer objects to be applied to the data.
            The first half of each tuple is the name of the transformer.

        n_jobs: int, optional.
            Number of jobs to run in parallel (default 1).
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        FrameMixin.__init__(self)

        self._feature_union = pipeline.FeatureUnion(
            transformer_list,
            n_jobs,
            transformer_weights)

    def fit(self, X, y=None):
        """
        Same signature as any sklearn step.
        """
        self._feature_union.fit(X, y)

        return self

    # Tmp Ami - get docstrings from sklearn.
    def fit_transform(self, X, y=None, **fit_params):
        """
        Same signature as any sklearn step.
        """
        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_fit_transform)(trans, name, w, X, y, **fit_params) for name, trans, w in self._transformers)
        return pd.concat(Xts, axis=1)

    def transform(self, X):
        """
        Same signature as any sklearn step.
        """
        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_transform)(trans, name, w, X) for name, trans, w in self._transformers)
        return pd.concat(Xts, axis=1)

    def get_params(self, deep=True):
        return self._feature_union.get_params(deep)

    def set_params(self, **params):
        return self._feature_union.set_params(**params)

    @property
    def transformer_list(self):
        return self._feature_union.transformer_list

    @property
    def _transformers(self):
        # Tmp Ami
        get_weight = lambda _: None
        return [(name, trans, get_weight(name)) for name, trans in self.transformer_list]

    @property
    def n_jobs(self):
        return self._feature_union.n_jobs

_FeatureUnion.__name__ = 'FeatureUnion'

for wrap in ['fit', 'transform', 'fit_transform']:
    try:
        functools.update_wrapper(
            getattr(_FeatureUnion, wrap),
            getattr(pipeline.FeatureUnion, wrap))
    except AttributeError:
        pass
