from __future__ import absolute_import

import functools

import pandas as pd
from sklearn import base
from sklearn import pipeline
from sklearn.externals import joblib

from ._frame_mixin import FrameMixin


def _fit(transformer, name, X):
    return transformer.transform(X)


def _transform(transformer, name, X):
    return transformer.transform(X)


def _fit_transform(transformer, name, X, y, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        return transformer.fit_transform(X, y, **fit_params)
    else:
        return transformer.fit(X, y, **fit_params).transform(X)


# Tmp Ami - take care of weights
class _FeatureUnion(base.BaseEstimator, base.TransformerMixin, FrameMixin):
    """
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

    Example:

        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1, 2, 3], 'b': [10, -3, 4]})

        >>> from ibex.sklearn import preprocessing as pd_preprocessing
        >>> from ibex.sklearn import pipeline as pd_pipeline

        >>> trn = pd_pipeline.FeatureUnion([
        ...     ('std', pd_preprocessing.StandardScaler()),
        ...     ('asb', pd_preprocessing.MaxAbsScaler())])
        >>> trn.fit_transform(X)
                a         b         a    b
        0 -1.224745  1.192166  0.333333  1.0
        1  0.000000 -1.254912  0.666667 -0.3
        2  1.224745  0.062746  1.000000  0.4

        >>> from ibex import trans
        >>>
        >>> trn = pd_preprocessing.StandardScaler() + pd_preprocessing.MaxAbsScaler()
        >>> trn.fit_transform(X)
                a         b         a    b
        0 -1.224745  1.192166  0.333333  1.0
        1  0.000000 -1.254912  0.666667 -0.3
        2  1.224745  0.062746  1.000000  0.4

        >>> trn = trans(pd_preprocessing.StandardScaler(), out_cols=['std_scale_a', 'std_scale_b'])
        >>> trn += trans(pd_preprocessing.MaxAbsScaler(), out_cols=['max_scale_a', 'max_scale_b'])
        >>> trn.fit_transform(X)
        std_scale_a  std_scale_b  max_scale_a  max_scale_b
        0    -1.224745     1.192166     0.333333          1.0
        1     0.000000    -1.254912     0.666667         -0.3
        2     1.224745     0.062746     1.000000          0.4
    """
    def __init__(self, transformer_list, n_jobs=1):
        FrameMixin.__init__(self)

        self._feature_union = pipeline.FeatureUnion(
            transformer_list,
            n_jobs)

    def fit(self, X, y=None):
        """
        Fits the transformer using ``X`` (and possibly ``y``).

        Returns:

            ``self``
        """
        self._feature_union.fit(X, y)

        return self

    # Tmp Ami - get docstrings from sklearn.
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the transformer using ``X`` (and possibly ``y``). Transforms
        ``X`` using the transformers, uses :func:`pandas.concat`
        to horizontally concatenate the results.

        Returns:

            ``self``
        """
        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_fit_transform)(trans, name, X, y, **fit_params) for name, trans, in self.transformer_list)
        return pd.concat(Xts, axis=1)

    def transform(self, X):
        """
        Transforms ``X`` using the transformers, uses :func:`pandas.concat`
        to horizontally concatenate the results.
        """
        Xts = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(_transform)(trans, name, X) for name, trans in self.transformer_list)
        return pd.concat(Xts, axis=1)

    def get_feature_names(self):
        return self._feature_union.get_feature_names()

    def get_params(self, deep=True):
        params = self._feature_union.get_params(deep)
        del params['transformer_weights']
        return params

    def set_params(self, **params):
        return self._feature_union.set_params(**params)

    @property
    def transformer_list(self):
        return self._feature_union.transformer_list

    @property
    def n_jobs(self):
        return self._feature_union.n_jobs

_FeatureUnion.__name__ = 'FeatureUnion'

_wrapped = [
    'get_feature_names',
    'get_params',
    'set_params',
]

for wrap in _wrapped:
    try:
        functools.update_wrapper(getattr(_FeatureUnion, wrap), getattr(pipeline.FeatureUnion, wrap))
    except AttributeError:
        print('wrap failed', wrap)
        pass
