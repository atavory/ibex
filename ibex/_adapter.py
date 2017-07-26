# # Auto generted from _adapter.py.jinja2

from __future__ import absolute_import

import inspect

import six
import numpy as np
import pandas as pd
from sklearn import pipeline

from ._frame_mixin import FrameMixin


def frame(step):
    """
    Arguments:
        ``step``: blah Returns:
        ``step`` :py:class:`ibex.sklearn.base.BaseEstimator` :py:class:`ibex.FrameMixin`

    Example:

        >>> from sklearn import linear_model
        >>> import ibex
        >>>
        >>> prd = ibex.frame(linear_model.LinearRegression())
        >>> prd
        Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        >>> PDLinearRegression = ibex.frame(linear_model.LinearRegression)
        >>> PDLinearRegression()
        Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
        >>> PDLinearRegression(fit_intercept=False)
        Adapter[LinearRegression](copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)
    """
    if isinstance(step, pipeline.Pipeline):
        return frame(pipeline.Pipeline)(steps=step.steps)

    if not inspect.isclass(step):
        params = step.get_params()
        f = frame(type(step))(**params)
        return f

    class _Adapter(step, FrameMixin):
        def __repr__(self):
            parts = step.__repr__(self).split('(', 1)
            return 'Adapter[' + parts[0] + '](' + parts[1]

        def __str__(self):
            return self.__repr__()

        def fit_transform(self, X, *args):
            return self.__run(super(_Adapter, self).fit_transform, 'fit_transform'.startswith('fit'), X, *args)
            
        def predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_proba, 'predict_proba'.startswith('fit'), X, *args)
            
        def sample_y(self, X, *args):
            return self.__run(super(_Adapter, self).sample_y, 'sample_y'.startswith('fit'), X, *args)
            
        def score_samples(self, X, *args):
            return self.__run(super(_Adapter, self).score_samples, 'score_samples'.startswith('fit'), X, *args)
            
        def staged_predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict_proba, 'staged_predict_proba'.startswith('fit'), X, *args)
            
        def apply(self, X, *args):
            return self.__run(super(_Adapter, self).apply, 'apply'.startswith('fit'), X, *args)
            
        def bic(self, X, *args):
            return self.__run(super(_Adapter, self).bic, 'bic'.startswith('fit'), X, *args)
            
        def perplexity(self, X, *args):
            return self.__run(super(_Adapter, self).perplexity, 'perplexity'.startswith('fit'), X, *args)
            
        def fit(self, X, *args):
            return self.__run(super(_Adapter, self).fit, 'fit'.startswith('fit'), X, *args)
            
        def decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).decision_function, 'decision_function'.startswith('fit'), X, *args)
            
        def aic(self, X, *args):
            return self.__run(super(_Adapter, self).aic, 'aic'.startswith('fit'), X, *args)
            
        def partial_fit(self, X, *args):
            return self.__run(super(_Adapter, self).partial_fit, 'partial_fit'.startswith('fit'), X, *args)
            
        def predict(self, X, *args):
            return self.__run(super(_Adapter, self).predict, 'predict'.startswith('fit'), X, *args)
            
        def radius_neighbors(self, X, *args):
            return self.__run(super(_Adapter, self).radius_neighbors, 'radius_neighbors'.startswith('fit'), X, *args)
            
        def staged_decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).staged_decision_function, 'staged_decision_function'.startswith('fit'), X, *args)
            
        def staged_predict(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict, 'staged_predict'.startswith('fit'), X, *args)
            
        def inverse_transform(self, X, *args):
            return self.__run(super(_Adapter, self).inverse_transform, 'inverse_transform'.startswith('fit'), X, *args)
            
        def fit_predict(self, X, *args):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict'.startswith('fit'), X, *args)
            
        def kneighbors(self, X, *args):
            return self.__run(super(_Adapter, self).kneighbors, 'kneighbors'.startswith('fit'), X, *args)
            
        def predict_log_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_log_proba, 'predict_log_proba'.startswith('fit'), X, *args)
            
        def transform(self, X, *args):
            return self.__run(super(_Adapter, self).transform, 'transform'.startswith('fit'), X, *args)
            
        
        def __run(self, fn, fit, X, *args):
            # Tmp Ami - why not in function adapter? where are uts?
            if fit:
                self.x_columns = X.columns

            res = fn(self.__x(X), *args)
            return self.__process_wrapped_call_res(X[self.x_columns], res)

        def score(self, X, y, sample_weights=None):
            self._no_output_pd = True
            try:
                return step.score(self, X, y, sample_weights)
            finally:
                self._no_output_pd = False

        # Tmp Ami - should be in base?
        def __x(self, X):
            # Tmp Ami - should be in base?
            X = X[self.x_columns]
            # Tmp Ami - is_subclass or isinstance?
            return X if FrameMixin.is_subclass(self) else X.as_matrix()

        def __process_wrapped_call_res(self, X, res):
            if hasattr(self, '_no_output_pd') and self._no_output_pd:
                return res

            if isinstance(res, np.ndarray):
                if len(res.shape) == 1:
                    return pd.Series(res, index=X.index)

                if len(res.shape) == 2:
                    if len(X.columns) == res.shape[1]:
                        columns = X.columns
                    else:
                        columns = [' ' for _ in range(res.shape[1])]
                    return pd.DataFrame(res, index=X.index, columns=columns)

            return res

        def __getattribute__(self, name):
            base_attr = super(_Adapter, self).__getattribute__(name)
            if name == 'feature_importances_':
                return pd.Series(base_attr, index=self.x_columns)
            return base_attr

    _Adapter.__name__ = step.__name__

    for name, func in vars(_Adapter).items():
        if name.startswith('_'):
            continue

        if not six.callable(func) and not func.__doc__:
            continue

        parfunc = getattr(step, name, None)
        if parfunc and getattr(parfunc, '__doc__', None):
            func.__doc__ = parfunc.__doc__

    wrapped = [
        'fit_transform',
        'predict_proba',
        'sample_y',
        'score_samples',
        'staged_predict_proba',
        'apply',
        'bic',
        'perplexity',
        'fit',
        'decision_function',
        'aic',
        'partial_fit',
        'predict',
        'radius_neighbors',
        'staged_decision_function',
        'staged_predict',
        'inverse_transform',
        'fit_predict',
        'kneighbors',
        'predict_log_proba',
        'transform',
        
    ]
    for wrap in wrapped:
        if not hasattr(step, wrap) and hasattr(_Adapter, wrap):
            delattr(_Adapter, wrap)

    return _Adapter