from __future__ import absolute_import

import operator

import pandas as pd
from sklearn import base
from sklearn import pipeline

from ._frame_mixin import FrameMixin


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

        transformer_weights: dict, optional.
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        FrameMixin.__init__(self)

        self._feature_union = pipeline.FeatureUnion(
            transformer_list,
            n_jobs,
            transformer_weights)

    # Tmp Ami - get docstrings from sklearn.
    def fit_transform(self, X, y=None):
        """
        Same signature as any sklearn step.
        """
        Xts = [e.fit_transform(X, y) for e in self._transformers]
        # Tmp Ami - add checks for indexes' equality. Add ut; add docs
        return pd.concat(
            [pd.DataFrame(Xt, index=X.index) for Xt in Xts],
            axis=1)

    def fit(self, X, y=None):
        """
        Same signature as any sklearn step.
        """
        self._feature_union.fit(
            X,
            y)

        return self

    def transform(self, X):
        """
        Same signature as any sklearn step.
        """
        Xts = [e.transform(X) for e in self._transformers]
        return pd.concat(
            [pd.DataFrame(Xt, index=X.index) for Xt in Xts],
            axis=1)

    def get_params(self, deep=True):
        return self._feature_union.get_params(deep)

    def set_params(self, **params):
        return self._feature_union.set_params(**params)

    @property
    def transformer_list(self):
        return self._feature_union.transformer_list

    @property
    def _transformers(self):
        return [operator.itemgetter(1)(e) for e in self.transformer_list]

_FeatureUnion.__name__ = 'FeatureUnion'
