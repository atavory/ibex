from __future__ import absolute_import

import functools
import inspect

import six
import numpy as np
import pandas as pd
from sklearn import pipeline

from ._frame_mixin import FrameMixin


def make_adapter(step):

    class _Adapter(step, FrameMixin):
        def __repr__(self):
            parts = step.__repr__(self).split('(', 1)
            return 'Adapter[' + parts[0] + '](' + parts[1]

        def __str__(self):
            return self.__repr__()

        def fit_transform(self, X, *args):
            return self.__run(super(_Adapter, self).fit_transform, 'fit_transform', X, *args)

        def predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_proba, 'predict_proba', X, *args)

        def sample_y(self, X, *args):
            return self.__run(super(_Adapter, self).sample_y, 'sample_y', X, *args)

        def score_samples(self, X, *args):
            return self.__run(super(_Adapter, self).score_samples, 'score_samples', X, *args)

        def staged_predict_proba(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict_proba, 'staged_predict_proba', X, *args)

        def apply(self, X, *args):
            return self.__run(super(_Adapter, self).apply, 'apply', X, *args)

        def bic(self, X, *args):
            return self.__run(super(_Adapter, self).bic, 'bic', X, *args)

        def perplexity(self, X, *args):
            return self.__run(super(_Adapter, self).perplexity, 'perplexity', X, *args)

        def fit(self, X, *args):
            return self.__run(super(_Adapter, self).fit, 'fit', X, *args)

        def decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).decision_function, 'decision_function', X, *args)

        def aic(self, X, *args):
            return self.__run(super(_Adapter, self).aic, 'aic', X, *args)

        def partial_fit(self, X, *args):
            return self.__run(super(_Adapter, self).partial_fit, 'partial_fit', X, *args)

        def predict(self, X, *args):
            return self.__run(super(_Adapter, self).predict, 'predict', X, *args)

        def radius_neighbors(self, X, *args):
            return self.__run(super(_Adapter, self).radius_neighbors, 'radius_neighbors', X, *args)

        def staged_decision_function(self, X, *args):
            return self.__run(super(_Adapter, self).staged_decision_function, 'staged_decision_function', X, *args)

        def staged_predict(self, X, *args):
            return self.__run(super(_Adapter, self).staged_predict, 'staged_predict', X, *args)

        def inverse_transform(self, X, *args):
            return self.__run(super(_Adapter, self).inverse_transform, 'inverse_transform', X, *args)

        def fit_predict(self, X, *args):
            return self.__run(super(_Adapter, self).fit_predict, 'fit_predict', X, *args)

        def kneighbors(self, X, *args):
            return self.__run(super(_Adapter, self).kneighbors, 'kneighbors', X, *args)

        def predict_log_proba(self, X, *args):
            return self.__run(super(_Adapter, self).predict_log_proba, 'predict_log_proba', X, *args)

        def transform(self, X, *args):
            return self.__run(super(_Adapter, self).transform, 'transform', X, *args)

        def score(self, X, *args):
            return self.__run(super(_Adapter, self).score, 'transform', X, *args)

        def __run(self, fn, name, X, *args):
            if hasattr(self, '_in_op') and self._in_op:
                return fn(X, *args)

                return super(_Adapter, self).decision_function(X, *args)
            # Tmp Ami - why not in function adapter? where are uts?
            if name.startswith('fit'):
                self.x_columns = X.columns

            base_attr = getattr(step, name)
            if six.PY3:
                params = list(inspect.signature(base_attr).parameters)
            else:
                params = inspect.getargspec(base_attr)[0]

            # Tmp Ami - write a ut for this; remove todo from docs
            if len(params) > 2 and params[2] == 'y' and len(args) > 0 and args[0] is not None:
                if not X.index.equals(args[0].index):
                    raise ValueError('Indexes do not match')

            self._in_op = True
            try:
                res = fn(self.__x(X), *args)
            finally:
                self._in_op = False

            return self.__process_wrapped_call_res(X[self.x_columns], res)

        # Tmp Ami - should be in base?
        def __x(self, X):
            X = X[self.x_columns]
            # Tmp Ami - erase is_subclass
            return X if FrameMixin.is_subclass(step) else X.as_matrix()

        def __process_wrapped_call_res(self, X, res):
            if hasattr(self, '_in_op') and self._in_op:
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

    return _Adapter

def frame(step):
    """
    Arguments:
        step: either a step class or a step object. The class (or class of the
            object) should subclass :py:class:`ibex.sklearn.base.BaseEstimator`.

    Returns:
        If ``step`` is a class, returns a class; if ``step`` is an object,
            returns an object. Note that the result will subclass ``step``
            and :py:class:`ibex.FrameMixin`

    Example:

        >>> from sklearn import linear_model
        >>> import ibex
        >>>

        We can use ``frame`` to adapt an object:

        >>> prd = ibex.frame(linear_model.LinearRegression())
        >>> prd
        Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

        We can use ``frame`` to adapt a class:

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

    _Adapter = make_adapter(step)

    _Adapter.__name__ = step.__name__

    for name, func in vars(_Adapter).items():
        if name.startswith('_'):
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
        elif six.callable(getattr(_Adapter, wrap)):
            try:
                functools.update_wrapper(getattr(_Adapter, wrap), getattr(step, wrap))
            except AttributeError:
                pass

    return _Adapter
