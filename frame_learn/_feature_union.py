import numpy as np
import pandas as pd
import sklearn
from sklearn import pipeline

from ._frame_mixin import FrameMixin


class FeatureUnion(FrameMixin):
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

    def fit_transform(self, x, y):
        """
        Same signature as any sklearn step.
        """
        xt = self._feature_union.fit_transform(
            x,
            y)

        return pd.DataFrame(xt, index=x.index)

    def fit(self, x, y):
        """
        Same signature as any sklearn step.
        """
        self._feature_union.fit(
            x,
            y)

        return self

    def transform(self, x):
        """
        Same signature as any sklearn step.
        """
        xt = self._feature_union.transform(x)

        return pd.DataFrame(xt, index=x.index)
